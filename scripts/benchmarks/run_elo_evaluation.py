import argparse
import math
from evaluate_orion import evaluate


def compute_elo(win_rate, baseline_rating=1200):
    """Compute approximate Elo rating based on win rate against a baseline opponent.
    Uses the standard Elo formula: R = R0 + 400 * log10(p / (1 - p))
    where R0 is the baseline opponent rating and p is the win probability.
    """
    if win_rate <= 0:
        return baseline_rating - 400 * 4  # very low
    if win_rate >= 1:
        return baseline_rating + 400 * 4  # very high
    return baseline_rating + 400 * math.log10(win_rate / (1 - win_rate))


def main():
    parser = argparse.ArgumentParser(description="Run ORION evaluation and compute approximate Elo rating.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play for evaluation")
    parser.add_argument("--orion-color", default="white", choices=["white", "black"], help="Color ORION will play as")
    parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint (optional)")
    args = parser.parse_args()

    # Run the existing evaluation function which returns printed results.
    # We'll capture the output by re‑using the evaluation logic directly.
    # For simplicity, we import the evaluate function and call it, but we also
    # replicate the result aggregation here to compute win rate.
    from evaluate_orion import play_game
    from orion.model.agent import ORIONAgent
    from orion.model.network import ORIONNetwork
    from orion.model.encoder import ChessEncoder
    from orion.utils.checkpoint import Checkpoint
    import random

    network = ORIONNetwork()
    if args.checkpoint:
        ckpt = Checkpoint(args.checkpoint)
        ckpt.load(network, args.checkpoint)
    encoder = ChessEncoder()
    agent = ORIONAgent(network, encoder, seed=42)

    wins, losses, draws = 0, 0, 0
    for i in range(args.games):
        result = play_game(agent, args.orion_color, seed=1000 + i)
        status = result["status"]
        if status.startswith("checkmate:"):
            winner = result["winner"]
            if winner == args.orion_color:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total if total else 0
    elo = compute_elo(win_rate)

    print("=== ORION ELO EVALUATION ===")
    print(f"Games played   : {total}")
    print(f"Wins           : {wins}")
    print(f"Losses         : {losses}")
    print(f"Draws          : {draws}")
    print(f"Win rate       : {win_rate:.3f}")
    print(f"Approx. Elo    : {elo:.0f}")

if __name__ == "__main__":
    main()
