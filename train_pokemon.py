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

from orion.pokemon.environment import PokemonEnv, NUM_ACTIONS, BUTTONS
from orion.pokemon.network import (
    PokemonResNet,
    PokemonEncoder,
    STATE_DIM,
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

    def _select_action(
        self, screen: torch.Tensor, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select action using the policy network."""
        with torch.no_grad():
            screen = screen.unsqueeze(0).to(self.device)
            state = state.unsqueeze(0).to(self.device)
            policy_logits, value = self.model(screen, state)

            probs = Categorical(logits=policy_logits)
            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = probs.sample()

            log_prob = probs.log_prob(action)

        return action.item(), log_prob.item(), value.item()

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

    def collect_rollout(self) -> RolloutBuffer:
        """Collect experience for one rollout."""
        buffer = RolloutBuffer()
        self.encoder.reset()

        obs, _ = self.env.reset()
        screen, state = self.encoder.encode(obs)

        episode_reward = 0.0
        episode_length = 0

        for step in range(self.rollout_length):
            action, log_prob, value = self._select_action(screen, state)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

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
            buffer.dones.append(float(done))

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.global_step += 1

            if done:
                # Get final value estimate
                with torch.no_grad():
                    final_screen, final_state = self.encoder.encode(obs)
                    s = final_screen.unsqueeze(0).to(self.device)
                    st = final_state.unsqueeze(0).to(self.device)
                    _, last_value = self.model(s, st)
                    last_value = last_value.item() if not terminated else 0.0

                # Record episode stats
                self.episode_count += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_badges.append(info.get("badges", 0))

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

                # Reset for next episode
                self.encoder.reset()
                obs, _ = self.env.reset()
                screen, state = self.encoder.encode(obs)
                episode_reward = 0.0
                episode_length = 0
            else:
                screen, state = self.encoder.encode(obs)
                # Bootstrap with value estimate of last state (not 0)
                with torch.no_grad():
                    s = screen.unsqueeze(0).to(self.device)
                    st = state.unsqueeze(0).to(self.device)
                    _, last_value = self.model(s, st)
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

        for rollout_idx in range(1, num_rollouts + 1):
            self._current_rollout = base_rollout + rollout_idx
            buffer = self.collect_rollout()
            metrics = self.update(buffer)

            elapsed = time.time() - start_time
            sps = self.global_step / elapsed if elapsed > 0 else 0

            if rollout_idx % log_interval == 0:
                avg_reward = (
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                )
                avg_length = (
                    np.mean(self.episode_lengths) if self.episode_lengths else 0
                )
                avg_badges = np.mean(self.episode_badges) if self.episode_badges else 0
                max_badges = max(self.episode_badges) if self.episode_badges else 0

                print(
                    f"Rollout {self._current_rollout:04d}/{base_rollout + num_rollouts} | "
                    f"Ep: {self.episode_count:05d} | "
                    f"Steps: {self.global_step:08d} | "
                    f"R: {avg_reward:8.1f} | "
                    f"Len: {avg_length:6.0f} | "
                    f"Badges: {avg_badges:.1f}/{max_badges} | "
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
                    "best_reward": self.best_reward,
                    "best_badges": self.best_badges,
                    "elapsed_s": elapsed,
                    **{k: v for k, v in metrics.items()},
                }
                with open(log_path, "w") as f:
                    json.dump(log, f, indent=2)

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
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
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
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--headless", action="store_true", default=True, help="Run headless"
    )
    args = parser.parse_args()

    # Create environment
    env = PokemonEnv(
        rom_path=args.rom,
        headless=args.headless,
        state_path=args.state,
        max_steps=args.max_steps,
    )

    # Create model and encoder
    model = PokemonResNet(num_buttons=NUM_ACTIONS)
    encoder = PokemonEncoder(frame_stack=4)

    # Create trainer
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
