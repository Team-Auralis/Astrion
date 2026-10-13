"""
PPO (Proximal Policy Optimization) training for Pokemon Gen 1 AI.

Usage:
    python train_pokemon.py --rom roms/pokemon_red.gb --episodes 1000
    python train_pokemon.py --rom roms/pokemon_red.gb --resume checkpoints/pokemon/ppo_latest.pt
"""

import argparse
import os
import sys
import time
import json
from collections import deque
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from orion.pokemon.environment import PokemonEnv
from orion.pokemon.network import (
    PokemonResNet,
    PokemonRecurrentNet,
    PokemonEncoder,
    SCREEN_H,
    SCREEN_W,
)


class RolloutBuffer:
    """Stores experience from one PPO rollout."""

    def __init__(self):
        self.observations: List[dict] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.returns: List[float] = []
        self.advantages: List[float] = []
        # Per-transition LSTM hidden state entering each step (recurrent mode)
        self.hidden_states: List = []
        # Per-transition flag recording whether the recurrent hidden state
        # entering this step was reset to fresh zeros (first step, episode end,
        # persistent-mode truncation, or hard reboot). Recurrent BPTT segments
        # must break on these boundaries even when the GAE `done` flag stays 0
        # (a persistent-mode max_steps truncation resets hx without marking the
        # transition terminal).
        self.hidden_resets: List[float] = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.actions)


class PPOTrainer:
    def __init__(
        self,
        env: PokemonEnv,
        model: PokemonResNet,
        encoder: PokemonEncoder,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.03,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        rollout_length: int = 2048,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints/pokemon",
        persist: bool = False,
        persist_max_steps: int = 400000,
        recurrent: bool = False,
        bptt_horizon: int = 16,
    ):
        self.env = env
        self.encoder = encoder
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.rollout_length = rollout_length
        self.checkpoint_dir = checkpoint_dir
        # In persistent mode the game keeps running across episode boundaries;
        # episodes are only bookkeeping windows. A hard cap forces a fresh
        # boot if the agent has wandered for a very long time.
        self.persist = persist
        self.persist_max_steps = persist_max_steps
        self._persist_game_steps = 0
        # Set by collect_rollout: the first rollout must boot a fresh game even
        # in persistent mode; later rollouts continue the running game.
        self._first_rollout = True
        self.recurrent = recurrent
        self.bptt_horizon = bptt_horizon

        # Safeguard: rollout must be >= max_steps so episodes actually complete
        max_steps = getattr(env, "max_steps", 2048)
        if self.rollout_length < max_steps:
            print(
                f"[WARN] rollout_length ({self.rollout_length}) < max_steps ({max_steps}). "
                f"Episodes would never complete. Auto-adjusting to {max_steps}."
            )
            self.rollout_length = max_steps

        # Adaptive entropy: raise coefficient if policy collapses
        self.min_entropy = 0.5  # Target minimum policy entropy (nats)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=100000,
        )

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training stats
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = -float("inf")
        self.best_badges = 0

        # Episode tracking
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        self.episode_badges: deque = deque(maxlen=100)
        self.episode_party: deque = deque(maxlen=100)
        self.episode_levels: deque = deque(maxlen=100)
        self.episode_maps: deque = deque(maxlen=100)

        # Live view: dump a color screenshot every `live_every` steps so the
        # dashboard can show the agent's real screen (movement, battles, ...).
        self.live_every = 100
        self.live_path = os.path.join(checkpoint_dir, "live.png")

    def _dump_live(self):
        """Save a color PNG of the current emulator screen for the dashboard."""
        try:
            from PIL import Image

            pyboy = getattr(self.env, "_pyboy", None)
            if pyboy is None:
                return
            arr = pyboy.screen.ndarray  # (144, 160, 4) RGBA
            img = Image.fromarray(arr[:, :, :3])
            tmp = self.live_path + ".tmp"
            img.save(tmp, "PNG")
            os.replace(tmp, self.live_path)
        except Exception:
            pass

    def _select_action(
        self,
        screen: torch.Tensor,
        state: torch.Tensor,
        hx: tuple = None,
        deterministic: bool = False,
    ):
        """Select action using the policy network."""
        with torch.no_grad():
            screen = screen.unsqueeze(0).to(self.device)
            state = state.unsqueeze(0).to(self.device)
            out = self.model(screen, state, hx)
            # Recurrent nets return ((logits,value), hx_new); FF nets return (logits,value)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
                (policy_logits, value), hx_new = out
            elif isinstance(out, tuple) and len(out) == 2:
                policy_logits, value = out
                hx_new = hx
            else:
                policy_logits, value = out
                hx_new = hx

            probs = Categorical(logits=policy_logits)
            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = probs.sample()

            log_prob = probs.log_prob(action)

        return action.item(), log_prob.item(), value.item(), hx_new

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
        last_value: float,
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def collect_rollout(self):
        """Collect experience for one rollout."""
        buffer = RolloutBuffer()
        self.encoder.reset()

        # Persistent mode: only the first rollout boots a fresh game; every
        # later rollout continues the game where the previous one left off.
        if self.persist and not self._first_rollout:
            obs = self.env.soft_reset()
        else:
            obs, _ = self.env.reset()
        self._first_rollout = False
        screen, state = self.encoder.encode(obs)

        episode_reward = 0.0
        episode_length = 0
        hx = None

        for step in range(self.rollout_length):
            res = self._select_action(screen, state, hx)
            action, log_prob, value, hx_new = res

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            episode_ended = terminated or truncated
            # In persistent mode a truncation is not terminal (the game keeps
            # running), so it must not cut the advantage recursion.
            done_for_gae = bool(terminated) if self.persist else bool(episode_ended)

            buffer.observations.append(
                {
                    "screen": obs["screen"].copy(),
                    "state": obs["state"].copy(),
                }
            )
            buffer.actions.append(action)
            buffer.rewards.append(reward)
            buffer.values.append(value)
            buffer.log_probs.append(log_prob)
            buffer.dones.append(float(done_for_gae))
            # Store the hidden state entering this step (recurrent BPTT) and
            # whether it is a fresh-reset boundary. hx is None at the first step
            # and right after every episode end; in persistent mode a max_steps
            # truncation resets hx while done_for_gae stays 0, so BPTT
            # segmentation needs this marker separate from `dones`.
            if self.recurrent:
                if hx is not None:
                    buffer.hidden_states.append((hx[0].cpu(), hx[1].cpu()))
                else:
                    buffer.hidden_states.append(None)
                buffer.hidden_resets.append(float(hx is None))

            obs = next_obs
            hx = hx_new
            episode_reward += reward
            episode_length += 1
            self.global_step += 1

            if self.global_step % self.live_every == 0:
                self._dump_live()

            if episode_ended:
                # Get final value estimate
                with torch.no_grad():
                    final_screen, final_state = self.encoder.encode(obs)
                    s = final_screen.unsqueeze(0).to(self.device)
                    st = final_state.unsqueeze(0).to(self.device)
                    out = self.model(s, st, hx)
                    if (
                        isinstance(out, tuple)
                        and len(out) == 2
                        and isinstance(out[0], tuple)
                    ):
                        (policy_logits, value), _ = out
                        last_value = value.item() if not terminated else 0.0
                    elif isinstance(out, tuple) and len(out) == 2:
                        _, last_value = out
                        last_value = last_value.item() if not terminated else 0.0
                    else:
                        _, last_value = out
                        last_value = last_value.item() if not terminated else 0.0

                # Record episode stats
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_badges.append(info.get("badges", 0))
                self.episode_party.append(info.get("party_count", 0))
                self.episode_levels.append(info.get("level", 0))
                self.episode_maps.append(info.get("map_id", 0))

                # Save best checkpoint based on badges first, then reward
                episode_badges = info.get("badges", 0)
                best_badges = max(self.episode_badges) if self.episode_badges else 0

                # Prioritize badge count over reward (badges = real progress)
                if episode_badges > self.best_badges or (
                    episode_badges == self.best_badges
                    and episode_reward > self.best_reward
                ):
                    self.best_reward = episode_reward
                    self.best_badges = episode_badges
                    self.save_checkpoint("ppo_best.pt")
                    if episode_badges > 0:
                        print(
                            f"\n  [NEW BEST] {episode_badges} badges, "
                            f"reward {episode_reward:.1f}"
                        )

                # Reset for next episode. In persistent mode the game keeps
                # running (soft_reset) so progress in the world is not thrown
                # away every max_steps decisions; a boot reset only happens on
                # completion or after the hard cap.
                self._persist_game_steps += episode_length
                if (
                    self.persist
                    and not terminated
                    and self._persist_game_steps < self.persist_max_steps
                ):
                    obs = self.env.soft_reset()
                    screen, state = self.encoder.encode(obs)
                    hx = None
                else:
                    # Hard reboot: mark the previous transition terminal so GAE
                    # does not bootstrap value across the discontinuity.
                    if self.persist and len(buffer.dones) > 0:
                        buffer.dones[-1] = 1.0
                    self._persist_game_steps = 0
                    self.encoder.reset()
                    obs, _ = self.env.reset()
                    screen, state = self.encoder.encode(obs)
                    hx = None
                episode_reward = 0.0
                episode_length = 0
            else:
                screen, state = self.encoder.encode(obs)
                # Bootstrap with value estimate of last state (not 0)
                with torch.no_grad():
                    s = screen.unsqueeze(0).to(self.device)
                    st = state.unsqueeze(0).to(self.device)
                    out = self.model(s, st, hx)
                    if (
                        isinstance(out, tuple)
                        and len(out) == 2
                        and isinstance(out[0], tuple)
                    ):
                        (_, last_value), _ = out
                    elif isinstance(out, tuple) and len(out) == 2:
                        _, last_value = out
                    else:
                        _, last_value = out
                    last_value = last_value.item()

        # Compute advantages
        advantages, returns = self._compute_gae(
            buffer.rewards, buffer.values, buffer.dones, last_value
        )
        buffer.advantages = advantages
        buffer.returns = returns

        return buffer

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO update over the collected rollout."""
        if self.recurrent:
            return self._update_recurrent(buffer)
        obs_screens = torch.stack(
            [torch.from_numpy(o["screen"]).float() for o in buffer.observations]
        ).to(self.device)
        obs_states = torch.stack(
            [torch.from_numpy(o["state"]).float() for o in buffer.observations]
        ).to(self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(
            self.device
        )
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32).to(
            self.device
        )
        returns = torch.tensor(buffer.returns, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(buffer.values, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        dataset_size = len(buffer)
        indices = np.arange(dataset_size)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_screens = obs_screens[mb_idx]
                mb_states = obs_states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]

                # Forward pass
                policy_logits, values = self.model(mb_screens, mb_states)
                values = values.squeeze(-1)

                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss = (
                    0.5
                    * torch.max(
                        (values - mb_returns) ** 2, (value_clipped - mb_returns) ** 2
                    ).mean()
                )

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Adaptive entropy: if entropy is too low, boost the bonus
                if entropy.item() < self.min_entropy:
                    adaptive_entropy_coef = self.entropy_coef * 5.0
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - adaptive_entropy_coef * entropy
                    )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.scheduler.step()

        n = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def _update_recurrent(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """PPO update for recurrent policies with truncated BPTT.

        The rollout is processed sequentially (no random shuffling) in ordered
        segments of `bptt_horizon`. Segments are cut at hidden-state reset
        boundaries (episode ends, persistent-mode truncations, reboots) so the
        LSTM is never fed a stale hidden state across a point where collection
        used fresh zeros. Each segment is run step-by-step carrying `hx` so
        gradients flow through the LSTM over the horizon (truncated BPTT).
        """
        dataset_size = len(buffer)
        hidden_states = getattr(buffer, "hidden_states", [None] * dataset_size)
        # Hidden-state reset boundaries: transitions whose entering hidden state
        # was reset to fresh zeros during collection. These must break BPTT
        # segments even when the GAE `done` flag is 0 (persistent-mode
        # truncation resets hx without marking the transition terminal).
        hidden_resets = getattr(buffer, "hidden_resets", [0.0] * dataset_size)

        # Normalize advantages
        adv = np.asarray(buffer.advantages, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        horizon = max(1, self.bptt_horizon)

        for epoch in range(self.ppo_epochs):
            start = 0
            while start < dataset_size:
                end = min(start + horizon, dataset_size)
                # Cut the segment at the next hidden-state reset (which covers
                # episode boundaries and reboots). A reset at t means collection
                # entered step t with fresh zeros, so the segment must start
                # there; the GAE `done` check is kept as a belt-and-braces guard
                # for buffers created before `hidden_resets` existed.
                seg_end = end
                for t in range(start + 1, end):
                    reset = t < len(hidden_resets) and hidden_resets[t] >= 0.5
                    if reset or buffer.dones[t - 1] >= 1.0:
                        seg_end = t
                        break
                if seg_end <= start:
                    seg_end = start + 1

                # Initial hidden state for this segment (None => fresh episode)
                hx0 = hidden_states[start] if start < len(hidden_states) else None
                if hx0 is not None:
                    hx0 = (hx0[0].to(self.device), hx0[1].to(self.device))

                # Run the segment step-by-step so gradients flow through the LSTM
                policy_losses = []
                value_losses = []
                # Raw per-step entropy (for reporting) and entropy weighted by
                # the (possibly adaptive) coefficient (for the loss term).
                entropies = []
                weighted_entropies = []
                hx = hx0
                for t in range(start, seg_end):
                    screen = (
                        torch.from_numpy(buffer.observations[t]["screen"])
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    state = (
                        torch.from_numpy(buffer.observations[t]["state"])
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    out, hx = self.model(screen, state, hx)
                    (policy_logits, values) = out
                    values = values.squeeze(-1)

                    action = torch.tensor(
                        [buffer.actions[t]], dtype=torch.long, device=self.device
                    )
                    old_log_prob = torch.tensor(
                        [buffer.log_probs[t]], dtype=torch.float32, device=self.device
                    )
                    advantage = torch.tensor(
                        [adv[t]], dtype=torch.float32, device=self.device
                    )
                    ret = torch.tensor(
                        [buffer.returns[t]], dtype=torch.float32, device=self.device
                    )
                    old_value = torch.tensor(
                        [buffer.values[t]], dtype=torch.float32, device=self.device
                    )

                    dist = Categorical(logits=policy_logits)
                    new_log_prob = dist.log_prob(action)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    surr1 = ratio * advantage
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * advantage
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = old_value + torch.clamp(
                        values - old_value, -self.clip_epsilon, self.clip_epsilon
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            (values - ret) ** 2, (value_clipped - ret) ** 2
                        ).mean()
                    )

                    entropy_coef = self.entropy_coef
                    if entropy.item() < self.min_entropy:
                        entropy_coef = self.entropy_coef * 5.0

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)
                    weighted_entropies.append(entropy * entropy_coef)
                    # Do NOT detach here: gradients must flow through hx across
                    # the whole segment. BPTT is truncated at segment boundaries
                    # because the next segment starts from the (graph-free)
                    # stored hidden state.

                mean_policy_loss = torch.stack(policy_losses).mean()
                mean_value_loss = torch.stack(value_losses).mean()
                # Report the raw policy entropy (same scale as the feed-forward
                # path); the adaptive coefficient only enters the loss term.
                mean_entropy = torch.stack(entropies).mean()
                loss = (
                    mean_policy_loss
                    + self.value_coef * mean_value_loss
                    - torch.stack(weighted_entropies).mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += mean_policy_loss.item()
                total_value_loss += mean_value_loss.item()
                total_entropy += mean_entropy.item()
                n_updates += 1

                start = seg_end

        self.scheduler.step()
        n = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "episode_count": self.episode_count,
                "rollout": getattr(self, "_current_rollout", 0),
                "best_reward": self.best_reward,
                "best_badges": self.best_badges,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> bool:
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        self.best_reward = ckpt.get("best_reward", -float("inf"))
        self.best_badges = ckpt.get("best_badges", 0)
        # Older checkpoints have no "rollout" key; episodes and rollouts are
        # 1:1 here (rollout_length == max_steps), so episode_count works.
        self._resume_rollout = ckpt.get("rollout", ckpt.get("episode_count", 0))
        print(f"[RESUME] Loaded checkpoint: {path} (step={self.global_step})")
        print(
            f"[RESUME] Best: {self.best_badges} badges, reward {self.best_reward:.1f}"
        )
        return True

    def train(self, num_rollouts: int = 1000, log_interval: int = 1):
        """Main training loop."""
        print("=" * 70)
        print("       ORION Pokemon Gen 1 AI - PPO Training")
        print("=" * 70)
        print(f"Device       : {self.device}")
        print(f"Parameters   : {self.model.parameter_count():,}")
        print(f"Rollout size : {self.rollout_length}")
        print(f"Batch size   : {self.minibatch_size}")
        print(f"Total steps  : {num_rollouts * self.rollout_length:,}")
        print(f"Checkpoints  : {self.checkpoint_dir}")
        print("=" * 70)

        start_time = time.time()

        # Absolute rollout numbering so resumed runs continue where they left
        # off (checkpoints store the last completed rollout).
        base_rollout = getattr(self, "_resume_rollout", 0)
        # global_step is restored on resume, so measure speed from this run's
        # starting point instead of the lifetime total.
        base_step = self.global_step

        for rollout_idx in range(1, num_rollouts + 1):
            self._current_rollout = base_rollout + rollout_idx
            buffer = self.collect_rollout()
            metrics = self.update(buffer)

            elapsed = time.time() - start_time
            sps = (self.global_step - base_step) / elapsed if elapsed > 0 else 0

            if rollout_idx % log_interval == 0:
                avg_reward = (
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                )
                avg_length = (
                    np.mean(self.episode_lengths) if self.episode_lengths else 0
                )
                avg_badges = np.mean(self.episode_badges) if self.episode_badges else 0
                max_badges = max(self.episode_badges) if self.episode_badges else 0
                max_party = max(self.episode_party) if self.episode_party else 0
                max_level = max(self.episode_levels) if self.episode_levels else 0
                max_map = max(self.episode_maps) if self.episode_maps else 0

                print(
                    f"Rollout {self._current_rollout:04d}/{base_rollout + num_rollouts} | "
                    f"Ep: {self.episode_count:05d} | "
                    f"Steps: {self.global_step:08d} | "
                    f"R: {avg_reward:8.1f} | "
                    f"Len: {avg_length:6.0f} | "
                    f"Badges: {avg_badges:.1f}/{max_badges} | "
                    f"Party: {max_party} | Lvl: {max_level} | Map: {max_map} | "
                    f"P: {metrics['policy_loss']:.4f} | "
                    f"V: {metrics['value_loss']:.4f} | "
                    f"H: {metrics['entropy']:.3f} | "
                    f"{sps:.0f} steps/s"
                )

            # Save periodic checkpoint
            if self._current_rollout % 10 == 0:
                self.save_checkpoint("ppo_latest.pt")

                # Save training log
                log_path = os.path.join(self.checkpoint_dir, "training_log.json")
                log = {
                    "rollout": self._current_rollout,
                    "episodes": self.episode_count,
                    "global_step": self.global_step,
                    "avg_reward": float(np.mean(self.episode_rewards))
                    if self.episode_rewards
                    else 0,
                    "avg_badges": float(np.mean(self.episode_badges))
                    if self.episode_badges
                    else 0,
                    "max_badges": int(max(self.episode_badges))
                    if self.episode_badges
                    else 0,
                    "max_party": int(max_party),
                    "max_level": int(max_level),
                    "max_map": int(max_map),
                    "best_reward": self.best_reward,
                    "best_badges": self.best_badges,
                    "elapsed_s": elapsed,
                    **{k: v for k, v in metrics.items()},
                }
                with open(log_path, "w") as f:
                    json.dump(log, f, indent=2)
                # Append-only history so the dashboard can plot real curves
                # even though training_log.json only holds the latest entry.
                with open(
                    os.path.join(self.checkpoint_dir, "training_history.jsonl"), "a"
                ) as f:
                    f.write(json.dumps(log) + "\n")

        # Final save
        self.save_checkpoint("ppo_final.pt")
        print(
            f"\n[COMPLETE] Training finished! {self.episode_count} episodes, {self.global_step:,} steps"
        )


def main():
    parser = argparse.ArgumentParser(description="Train ORION Pokemon AI with PPO")
    parser.add_argument("--rom", required=True, help="Path to Pokemon ROM (.gb)")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of PPO rollouts"
    )
    parser.add_argument(
        "--rollout-length", type=int, default=2048, help="Steps per rollout"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (default: 0.99, or 0.998 with --recurrent)",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.02, help="Entropy bonus coefficient"
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon"
    )
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument(
        "--max-steps", type=int, default=2048, help="Max steps per episode"
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/pokemon", help="Checkpoint directory"
    )
    parser.add_argument("--resume", default=None, help="Resume from checkpoint file")
    parser.add_argument("--state", default=None, help="Initial save state path")
    parser.add_argument(
        "--curriculum",
        choices=("bedroom", "pallet", "rival"),
        default=None,
        help=(
            "Gen 1 curriculum start: resolve states/red_<stage>.state into "
            "--state (recorder: python record_intro_state.py <rom>; the "
            "'rival' state must be recorded manually)"
        ),
    )
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--headless", action="store_true", default=True, help="Run headless"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=8,
        help="Emulated frames per decision (higher = more game time per step)",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Keep the game running across episodes instead of rebooting",
    )
    parser.add_argument(
        "--persist-max-steps",
        type=int,
        default=400000,
        help="Hard cap on game steps before a persistent run reboots",
    )
    parser.add_argument(
        "--pokerl",
        action="store_true",
        help="PokeRL-style input hygiene: drop Start/Select, tap movement buttons",
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Use recurrent LSTM policy",
    )
    parser.add_argument(
        "--bptt-horizon",
        type=int,
        default=16,
        help="BPTT horizon for recurrent PPO",
    )
    args = parser.parse_args()

    # Recurrent PPO benefits from a longer horizon (winning stacks use
    # gamma=0.998); feed-forward keeps the original 0.99 default. An explicit
    # --gamma always wins.
    if args.gamma is None:
        args.gamma = 0.998 if args.recurrent else 0.99

    # ---- curriculum resolution (Gen 1) ----------------------------------
    # --curriculum <stage> resolves states/red_<stage>.state into the
    # effective --state. Never silently falls back: a missing state is an
    # actionable error (the 'rival' state is manual-only by design).
    if args.curriculum is not None:
        if args.state is not None:
            parser.error("--curriculum and --state are mutually exclusive")
        state_file = os.path.join("states", f"red_{args.curriculum}.state")
        if not os.path.isfile(state_file):
            if args.curriculum == "rival":
                print(
                    f"[curriculum] ERROR: state file not found: {state_file}\n"
                    "  The rival start must be recorded manually (no auto-scripting); "
                    "play through to the in-lab rival encounter, then save the state "
                    "and place it at states/red_rival.state.\n"
                    "  Recorded stages: 'bedroom' and 'pallet' via "
                    "`python record_intro_state.py <rom.gb>`.",
                    flush=True,
                )
            else:
                print(
                    f"[curriculum] ERROR: state file not found: {state_file}\n"
                    "  Record the Gen 1 curriculum states first:\n"
                    "    python record_intro_state.py <rom.gb> "
                    "[--out-dir states]",
                    flush=True,
                )
            sys.exit(2)
        args.state = state_file
        print(f"[curriculum] stage '{args.curriculum}' -> {state_file}", flush=True)

    # Create environment (auto-detect Gen 2 by cartridge family in the ROM
    # file name; supervisor passes full paths like "Pokemon - Gold ... .gbc").
    from orion.pokemon.gen2_env import PokemonGen2Env

    rom_lower = args.rom.lower()
    is_gen2 = any(
        tag in rom_lower for tag in ("gold", "silver", "crystal", "gen2", "g/s")
    )
    env_cls = PokemonGen2Env if is_gen2 else PokemonEnv
    if is_gen2:
        print(f"[gen2] detected Gen 2 ROM ({os.path.basename(args.rom)})")
        if args.curriculum is not None:
            print(
                "[curriculum] ERROR: --curriculum states are Gen 1 only "
                "(recorded from Pokemon Red); refusing to load them into a "
                "Gen 2 ROM.",
                flush=True,
            )
            sys.exit(2)

    # Create environment. --pokerl is a Gen 1 experiment (PokeRL-style action
    # space: 6 buttons, tapped movement); PokemonGen2Env does not expose the
    # flag yet, so it is only forwarded for Gen 1.
    if args.pokerl and is_gen2:
        print("[warn] --pokerl only applies to Gen 1 ROMs; ignoring it")
    env = env_cls(
        rom_path=args.rom,
        headless=args.headless,
        state_path=args.state,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        persistent=args.persistent,
        **({"pokerl_mode": True} if args.pokerl and not is_gen2 else {}),
    )

    # Create model and encoder. Use the env's actual action count so both the
    # default 8-button and --pokerl 6-button spaces produce a matching
    # policy head, and the env's state size so the pokerl 30-dim coverage
    # observation (and the default 20-dim one) both feed the state MLP.
    if args.recurrent:
        model = PokemonRecurrentNet(
            num_buttons=env.action_space.n, state_dim=env.state_feature_size
        )
    else:
        model = PokemonResNet(
            num_buttons=env.action_space.n, state_dim=env.state_feature_size
        )
    encoder = PokemonEncoder(frame_stack=4)

    # Create trainer
    # For recurrent runs, honor args.gamma (default 0.99 for FF; allow passing 0.998)
    trainer = PPOTrainer(
        env=env,
        model=model,
        encoder=encoder,
        lr=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.batch_size,
        rollout_length=args.rollout_length,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        persist=args.persistent,
        persist_max_steps=args.persist_max_steps,
        recurrent=args.recurrent,
        bptt_horizon=args.bptt_horizon,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train(num_rollouts=args.episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main()
