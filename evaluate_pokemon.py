"""
Evaluate a trained Pokemon AI checkpoint.

Usage:
    python evaluate_pokemon.py --rom roms/pokemon_red.gb --checkpoint checkpoints/pokemon/ppo_best.pt
"""

import argparse
import os
import time
import json

import numpy as np
import torch

from orion.pokemon.environment import PokemonEnv, NUM_ACTIONS, BUTTONS
from orion.pokemon.network import PokemonResNet, PokemonEncoder


def evaluate(
    rom_path: str,
    checkpoint_path: str,
    episodes: int = 10,
    max_steps: int = 4096,
    deterministic: bool = True,
    render: bool = False,
    state_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = PokemonEnv(
        rom_path=rom_path,
        headless=not render,
        render_mode="human" if render else None,
        state_path=state_path,
        max_steps=max_steps,
    )

    model = PokemonResNet(num_buttons=NUM_ACTIONS).to(device)
    encoder = PokemonEncoder(frame_stack=4)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[EVAL] Loaded checkpoint: {checkpoint_path}")
        print(
            f"[EVAL] Episode: {ckpt.get('episode_count', '?')}, Best reward: {ckpt.get('best_reward', '?')}"
        )
    else:
        print(
            f"[EVAL] WARNING: Checkpoint not found: {checkpoint_path} - using random weights"
        )

    model.eval()

    results = []
    action_distribution = [0] * NUM_ACTIONS

    for ep in range(episodes):
        encoder.reset()
        obs, _ = env.reset()
        screen, state = encoder.encode(obs)

        total_reward = 0.0
        total_steps = 0
        badges_max = 0
        maps_visited = set()
        done = False

        t0 = time.time()

        while not done:
            with torch.no_grad():
                s = screen.unsqueeze(0).to(device)
                st = state.unsqueeze(0).to(device)
                policy_logits, value = model(s, st)

                if deterministic:
                    action = policy_logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(policy_logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_steps += 1
            action_distribution[action] += 1
            badges_max = max(badges_max, info.get("badges", 0))
            maps_visited.add(info.get("map_id", 0))

            screen, state = encoder.encode(obs)

        elapsed = time.time() - t0

        results.append(
            {
                "episode": ep + 1,
                "reward": total_reward,
                "steps": total_steps,
                "badges": badges_max,
                "maps_visited": len(maps_visited),
                "time_s": elapsed,
                "final_level": info.get("level", 0),
                "final_party": info.get("party_count", 0),
            }
        )

        print(
            f"Episode {ep + 1:03d} | "
            f"Reward: {total_reward:8.1f} | "
            f"Steps: {total_steps:5d} | "
            f"Badges: {badges_max} | "
            f"Maps: {len(maps_visited):3d} | "
            f"Lv: {info.get('level', 0):3d} | "
            f"Party: {info.get('party_count', 0)} | "
            f"Time: {elapsed:.1f}s"
        )

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    rewards = [r["reward"] for r in results]
    badges = [r["badges"] for r in results]
    maps = [r["maps_visited"] for r in results]
    levels = [r["final_level"] for r in results]
    parties = [r["final_party"] for r in results]

    print(f"Episodes          : {len(results)}")
    print(f"Mean Reward       : {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"Max Badge Count   : {max(badges)} / 8")
    print(f"Mean Badges       : {np.mean(badges):.2f}")
    print(f"Mean Maps Visited : {np.mean(maps):.1f}")
    print(f"Mean Final Level  : {np.mean(levels):.1f}")
    print(f"Mean Party Size   : {np.mean(parties):.2f} / 6")
    print(f"Mean Steps        : {np.mean([r['steps'] for r in results]):.0f}")

    total_actions = sum(action_distribution)
    if total_actions > 0:
        print(f"\nAction Distribution:")
        for i, (btn, count) in enumerate(zip(BUTTONS, action_distribution)):
            pct = count / total_actions * 100
            bar = "#" * int(pct / 2)
            print(f"  {btn:8s}: {pct:5.1f}% {bar}")

    # Save results
    output_path = os.path.join(
        os.path.dirname(checkpoint_path) or ".", "evaluation_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(
            {
                "results": results,
                "summary": {
                    "mean_reward": float(np.mean(rewards)),
                    "max_badges": int(max(badges)),
                    "mean_badges": float(np.mean(badges)),
                    "mean_maps": float(np.mean(maps)),
                    "mean_level": float(np.mean(levels)),
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ORION Pokemon AI")
    parser.add_argument("--rom", required=True, help="Path to Pokemon ROM")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--max-steps", type=int, default=4096, help="Max steps per episode"
    )
    parser.add_argument("--state", default=None, help="Initial save state")
    parser.add_argument(
        "--stochastic", action="store_true", help="Sample actions instead of argmax"
    )
    parser.add_argument("--render", action="store_true", help="Show emulator window")
    args = parser.parse_args()

    evaluate(
        rom_path=args.rom,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        render=args.render,
        state_path=args.state,
    )


if __name__ == "__main__":
    main()
