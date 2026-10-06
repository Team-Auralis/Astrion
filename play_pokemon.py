"""
Watch a trained Pokemon AI play the game.

Usage:
    # Watch with emulator window
    python play_pokemon.py --rom roms/pokemon_red.gb --checkpoint checkpoints/pokemon/ppo_best.pt

    # Watch with random weights (untrained)
    python play_pokemon.py --rom roms/pokemon_red.gb

    # Watch on Gen 2
    python play_pokemon.py --rom roms/pokemon_gold.gbc --gen2 --checkpoint checkpoints/pokemon/gen2_best.pt

    # Record to video
    python play_pokemon.py --rom roms/pokemon_red.gb --record output.mp4
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def find_best_checkpoint(checkpoint_dir: str = "checkpoints/pokemon") -> str:
    """Auto-discover the best checkpoint in a directory."""
    candidates = [
        "ppo_best.pt",
        "ppo_final.pt",
        "ppo_latest.pt",
    ]

    for name in candidates:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return path

    # Search recursively
    for root, dirs, files in os.walk(checkpoint_dir):
        for f in sorted(files):
            if f.endswith(".pt"):
                return os.path.join(root, f)

    return None


def play(
    rom_path: str,
    checkpoint_path: str = None,
    gen2: bool = False,
    max_steps: int = 10000,
    deterministic: bool = True,
    speed: float = 0,  # 0 = unlimited, 1.0 = normal speed
    fps: int = 60,
):
    """
    Run the trained AI on a Pokemon game with visible emulator window.
    """
    from orion.pokemon.network import PokemonResNet, PokemonEncoder, NUM_ACTIONS

    # Auto-discover checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint()
        if checkpoint_path:
            print(f"[AUTO] Found checkpoint: {checkpoint_path}")
        else:
            print("[WARN] No checkpoint found - using random weights (untrained AI)")

    # Select environment
    if gen2:
        from orion.pokemon.gen2_env import PokemonGen2Env as EnvClass

        print("[GEN2] Using Gold/Silver/Crystal environment")
    else:
        from orion.pokemon.environment import PokemonEnv as EnvClass

        print("[GEN1] Using Red/Blue/Yellow environment")

    # Create environment (headless=False shows the emulator window)
    env = EnvClass(
        rom_path=rom_path,
        headless=False,  # Show window
        render_mode="human",
        max_steps=max_steps,
    )

    # Create model
    model = PokemonResNet(num_buttons=NUM_ACTIONS)
    encoder = PokemonEncoder(frame_stack=4)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print(f"[LOAD] Loaded checkpoint: {checkpoint_path}")
        print(f"[LOAD] Episodes trained: {ckpt.get('episode_count', '?')}")
        print(f"[LOAD] Best reward: {ckpt.get('best_reward', '?')}")
    else:
        device = torch.device("cpu")
        print("[WARN] Using random weights")

    model.eval()

    # Statistics
    total_reward = 0.0
    action_counts = [0] * NUM_ACTIONS
    action_names = ["A", "B", "Start", "Select", "Up", "Down", "Left", "Right"]

    print("\n" + "=" * 60)
    print("  ORION Pokemon AI - Playing")
    print("=" * 60)
    print(f"  ROM:         {rom_path}")
    print(f"  Checkpoint:  {checkpoint_path or 'None (random)'}")
    print(f"  Max steps:   {max_steps}")
    print(f"  Mode:        {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"  Speed:       {'Unlimited' if speed == 0 else f'{speed}x'}")
    print("=" * 60)
    print("\n  Watch the emulator window!")
    print("  Press Ctrl+C to stop.\n")

    encoder.reset()
    obs, _ = env.reset()
    screen, state = encoder.encode(obs)

    step = 0
    start_time = time.time()
    frame_delay = 1.0 / fps if fps > 0 else 0

    try:
        while step < max_steps:
            # Select action
            with torch.no_grad():
                s = screen.unsqueeze(0).to(device)
                st = state.unsqueeze(0).to(device)
                policy_logits, value = model(s, st)

                if deterministic:
                    action = policy_logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(policy_logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            action_counts[action] += 1
            step += 1

            # Update encoder
            screen, state = encoder.encode(obs)

            # Print progress every 100 steps
            if step % 100 == 0:
                elapsed = time.time() - start_time
                sps = step / elapsed if elapsed > 0 else 0
                badges = info.get("badges", 0)
                level = info.get("level", 0)
                maps = info.get("visited_maps", 0)

                # Show top 3 likely actions
                probs = torch.softmax(policy_logits, dim=-1).squeeze()
                top3 = torch.topk(probs, 3)
                top3_str = ", ".join(
                    f"{action_names[i]}({p:.2f})"
                    for i, p in zip(top3.indices.tolist(), top3.values.tolist())
                )

                print(
                    f"  Step {step:5d} | "
                    f"Reward: {total_reward:8.1f} | "
                    f"Badges: {badges} | "
                    f"Lv: {level:3d} | "
                    f"Maps: {maps:3d} | "
                    f"Top: {top3_str} | "
                    f"{sps:.1f} steps/s"
                )

            if done:
                print(f"\n[END] Episode finished at step {step}")
                if info.get("badges", 0) >= (16 if gen2 else 8):
                    print("[WIN] All badges earned!")
                break

            # Speed control
            if speed > 0:
                time.sleep(frame_delay / speed)

    except KeyboardInterrupt:
        print("\n[STOP] Stopped by user")

    # Final statistics
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Steps taken:     {step}")
    print(f"  Total reward:    {total_reward:.1f}")
    print(f"  Time elapsed:    {elapsed:.1f}s")
    print(f"  Average speed:   {step / elapsed:.1f} steps/s")
    print(f"  Final badges:    {info.get('badges', 0)}")
    print(f"  Final level:     {info.get('level', 0)}")
    print(f"  Maps visited:    {info.get('visited_maps', 0)}")
    print(f"  Party size:      {info.get('party_count', 0)}")

    print(f"\n  Action Distribution:")
    total_actions = sum(action_counts)
    for name, count in zip(action_names, action_counts):
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {name:7s}: {pct:5.1f}% {bar}")

    env.close()
    return total_reward, step


def main():
    parser = argparse.ArgumentParser(description="Watch ORION Pokemon AI play")
    parser.add_argument("--rom", required=True, help="Path to Pokemon ROM")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint (auto-discover if not set)",
    )
    parser.add_argument(
        "--gen2",
        action="store_true",
        help="Use Gen 2 (Gold/Silver/Crystal) environment",
    )
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps")
    parser.add_argument(
        "--stochastic", action="store_true", help="Sample actions instead of argmax"
    )
    parser.add_argument(
        "--speed", type=float, default=0, help="Playback speed (0=unlimited, 1=normal)"
    )
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    args = parser.parse_args()

    play(
        rom_path=args.rom,
        checkpoint_path=args.checkpoint,
        gen2=args.gen2,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        speed=args.speed,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
