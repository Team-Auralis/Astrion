"""Smoke test for the Pokemon training pipeline (no ROM required)."""

import numpy as np
import torch
from torch.distributions import Categorical

from orion.pokemon.network import (
    PokemonResNet,
    PokemonEncoder,
    NUM_BUTTONS,
    SCREEN_H,
    SCREEN_W,
    STATE_DIM,
)


def test_model():
    model = PokemonResNet(num_buttons=NUM_BUTTONS)
    print(f"PokemonResNet parameters: {model.parameter_count():,}")

    batch = 8
    screens = torch.randn(batch, 4, SCREEN_H, SCREEN_W)
    states = torch.randn(batch, STATE_DIM)
    policy, value = model(screens, states)
    assert policy.shape == (batch, NUM_BUTTONS), policy.shape
    assert value.shape == (batch, 1), value.shape
    print("Forward pass OK")


def test_encoder():
    enc = PokemonEncoder(frame_stack=4)
    enc.reset()
    obs = {
        "screen": np.random.randint(0, 255, (SCREEN_H, SCREEN_W), dtype=np.uint8),
        "state": np.random.randn(STATE_DIM).astype(np.float32),
    }
    s, st = enc.encode(obs)
    assert s.shape == (4, SCREEN_H, SCREEN_W), s.shape
    assert st.shape == (STATE_DIM,), st.shape
    # Stack multiple frames
    for _ in range(6):
        enc.encode(obs)
    s2, _ = enc.encode(obs)
    assert s2.shape == (4, SCREEN_H, SCREEN_W)
    print("Encoder OK")


def test_ppo_update():
    model = PokemonResNet(num_buttons=NUM_BUTTONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch = 32
    screens = torch.randn(batch, 4, SCREEN_H, SCREEN_W)
    states = torch.randn(batch, STATE_DIM)
    actions = torch.randint(0, NUM_BUTTONS, (batch,))
    old_log_probs = torch.randn(batch)
    advantages = torch.randn(batch)
    returns = torch.randn(batch)

    policy_logits, values = model(screens, states)
    values = values.squeeze(-1)

    dist = Categorical(logits=policy_logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = 0.5 * ((values - returns) ** 2).mean()
    loss = policy_loss + 0.5 * value_loss - 0.02 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"PPO update OK: P={policy_loss.item():.4f} V={value_loss.item():.4f} H={entropy.item():.4f}"
    )


def test_gae():
    def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    adv = compute_gae([1.0, 0.5, -0.1, 2.0], [0.5, 0.6, 0.4, 0.7], [0, 0, 0, 1], 0.0)
    assert len(adv) == 4
    print(f"GAE OK: {[f'{a:.3f}' for a in adv]}")


def test_rollout_buffer():
    import sys

    sys.path.insert(0, ".")
    from train_pokemon import RolloutBuffer

    buf = RolloutBuffer()
    assert len(buf) == 0
    buf.clear()
    print("RolloutBuffer OK")


if __name__ == "__main__":
    test_model()
    test_encoder()
    test_ppo_update()
    test_gae()
    test_rollout_buffer()
    print("\nALL POKEMON PIPELINE TESTS PASSED")
