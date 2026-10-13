import argparse
import random
from collections import Counter
from pathlib import Path

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.agent import ORIONAgent
from orion.utils.checkpoint import Checkpoint


def random_move(environment, rng):
    moves = environment.legal_moves()

    if not moves:
        return None

    return rng.choice(moves)


def play_game(orion, orion_color, seed):
    game = ChessEnvironment()
    rng = random.Random(seed)

    while True:
        status = game.game_status()

        if status not in ("playing", "check"):
            break

        if game.turn == orion_color:
            move = orion.choose_move(
                game,
                temperature=0
            )
        else:
            move = random_move(
                game,
                rng
            )

        if move is None:
            break

        if not game.make_move(move):
            raise RuntimeError(
                f"Illegal move returned by ORION: {move}"
            )

    status = game.game_status()

    winner = (
        status.split(":")[1]
        if status.startswith("checkmate:")
        else None
    )

    return {
        "status": status,
        "winner": winner,
        "plies": game.move_count
    }


def run_side(
    orion,
    orion_color,
    games,
    seed
):
    results = []

    for i in range(games):
        results.append(
            play_game(
                orion,
                orion_color,
                seed + i
            )
        )

    return results


def summarize(results, orion_color):
    wins = 0
    losses = 0
    draws = 0

    reasons = Counter()
    total_plies = 0

    for result in results:
        total_plies += result["plies"]

        if result["status"].startswith("checkmate:"):

            if result["winner"] == orion_color:
                wins += 1
            else:
                losses += 1

            reasons["checkmate"] += 1

        else:
            draws += 1
            reasons[result["status"]] += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (
            wins + 0.5 * draws
        ) / len(results),
        "average_plies": (
            total_plies / len(results)
        ),
        "reasons": reasons
    }


def evaluate(checkpoint_path):

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    network = ORIONNetwork()

    episode = Checkpoint(
        directory=path.parent
    ).load(
        network,
        path
    )

    encoder = ChessEncoder()

    orion = ORIONAgent(
        network,
        encoder,
        seed=42
    )

    white_results = run_side(
        orion=orion,
        orion_color="white",
        games=10,
        seed=1000
    )

    black_results = run_side(
        orion=orion,
        orion_color="black",
        games=10,
        seed=2000
    )

    white = summarize(
        white_results,
        "white"
    )

    black = summarize(
        black_results,
        "black"
    )

    all_results = []

    for result in white_results:
        all_results.append(
            (result, "white")
        )

    for result in black_results:
        all_results.append(
            (result, "black")
        )

    wins = 0
    losses = 0
    draws = 0
    total_plies = 0
    reasons = Counter()

    for result, color in all_results:

        total_plies += result["plies"]

        if result["status"].startswith("checkmate:"):

            if result["winner"] == color:
                wins += 1
            else:
                losses += 1

            reasons["checkmate"] += 1

        else:

            draws += 1
            reasons[result["status"]] += 1

    total_games = len(all_results)

    score = (
        wins + 0.5 * draws
    ) / total_games

    print("================================")
    print("     ORION v0.5 BENCHMARK")
    print("================================")
    print()
    print("Checkpoint     :", path)
    print("Episode        :", episode)
    print("Opponent       : fixed random")
    print("Total games    :", total_games)
    print()

    print("WHITE")
    print("  Wins         :", white["wins"])
    print("  Losses       :", white["losses"])
    print("  Draws        :", white["draws"])
    print("  Score        :", round(white["score"], 3))
    print("  Avg plies    :", round(white["average_plies"], 2))
    print()

    print("BLACK")
    print("  Wins         :", black["wins"])
    print("  Losses       :", black["losses"])
    print("  Draws        :", black["draws"])
    print("  Score        :", round(black["score"], 3))
    print("  Avg plies    :", round(black["average_plies"], 2))
    print()

    print("COMBINED")
    print("  Wins         :", wins)
    print("  Losses       :", losses)
    print("  Draws        :", draws)
    print("  Win rate     :", round(wins / total_games, 3))
    print("  Loss rate    :", round(losses / total_games, 3))
    print("  Draw rate    :", round(draws / total_games, 3))
    print("  Chess score  :", round(score, 3))
    print("  Avg plies    :", round(total_plies / total_games, 2))
    print()

    print("Termination reasons:")

    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    print()
    print("ORION v0.5 BENCHMARK COMPLETE")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "checkpoint",
        help="Path to ORION checkpoint"
    )

    args = parser.parse_args()

    evaluate(
        args.checkpoint
    )
