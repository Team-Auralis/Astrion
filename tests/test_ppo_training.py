"""End-to-end test of PPO training with mock Pokemon environment (no ROM)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orion.pokemon.mock_env import MockPokemonEnv
from orion.pokemon.network import PokemonResNet, PokemonEncoder, NUM_BUTTONS
from train_pokemon import PPOTrainer


def main():
    print("=" * 60)
    print("  PPO Training Pipeline Test (Mock Environment)")
    print("=" * 60)

    # Create mock environment
    env = MockPokemonEnv(max_steps=256, seed=42)

    # Verify observation space
    obs, _ = env.reset(seed=42)
    print(f"Screen shape: {obs['screen'].shape}")
    print(f"State shape:  {obs['state'].shape}")
    assert obs["screen"].shape == (72, 80), obs["screen"].shape
    assert obs["state"].shape == (20,), obs["state"].shape

    # Verify action space
    assert env.action_space.n == 8

    # Take some random steps
    total_reward = 0
    for i in range(50):
        action = i % 8
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    print(
        f"50 steps completed: reward={total_reward:.1f}, badges={info['badges']}, level={info['level']}"
    )
    assert info["badges"] >= 0

    # Create model and encoder
    model = PokemonResNet(num_buttons=NUM_BUTTONS)
    encoder = PokemonEncoder(frame_stack=4)

    # Create trainer with small rollout for quick test
    trainer = PPOTrainer(
        env=env,
        model=model,
        encoder=encoder,
        lr=3e-4,
        rollout_length=128,
        minibatch_size=32,
        ppo_epochs=2,
        checkpoint_dir="checkpoints/pokemon_test",
    )

    # Run 3 rollouts
    print("\nRunning 3 PPO rollouts...")
    trainer.train(num_rollouts=3, log_interval=1)

    # Verify checkpoint was saved (ppo_final.pt is always written by train())
    assert os.path.exists("checkpoints/pokemon_test/ppo_final.pt"), (
        "Final checkpoint not saved"
    )
    # ppo_latest.pt is only written every 10 rollouts, so force one here
    trainer.save_checkpoint("ppo_latest.pt")
    assert os.path.exists("checkpoints/pokemon_test/ppo_latest.pt"), (
        "Checkpoint not saved"
    )
    print("\nCheckpoints saved!")

    # Test loading
    fresh_model = PokemonResNet(num_buttons=NUM_BUTTONS)
    import torch

    ckpt = torch.load("checkpoints/pokemon_test/ppo_latest.pt", weights_only=False)
    fresh_model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Checkpoint loaded: step={ckpt['global_step']}, episodes={ckpt['episode_count']}"
    )

    # Verify inference works
    fresh_model.eval()
    with torch.no_grad():
        import torch as t

        s = t.randn(1, 4, 72, 80)
        st = t.randn(1, 20)
        policy, value = fresh_model(s, st)
        action = policy.argmax(dim=-1).item()
        print(
            f"Inference: action={action} ({['a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'][action]}), value={value.item():.3f}"
        )

    # Cleanup
    import shutil

    shutil.rmtree("checkpoints/pokemon_test", ignore_errors=True)

    print("\n" + "=" * 60)
    print("  ALL PPO TRAINING TESTS PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Place your Pokemon ROM at: roms/pokemon_red.gb")
    print("  2. Run: python train_pokemon.py --rom roms/pokemon_red.gb --episodes 1000")
    print(
        "  3. Evaluate: python evaluate_pokemon.py --rom roms/pokemon_red.gb --checkpoint checkpoints/pokemon/ppo_best.pt"
    )


if __name__ == "__main__":
    main()
