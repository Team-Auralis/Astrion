import json
import math
import sys

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.utils.checkpoint import Checkpoint


DATASET = "lichess_1000.json"
CHECKPOINT_PATH = "checkpoints/v0.5.1_supervised/orion_5.json"


def uci_to_move(uci):
    if len(uci) < 4:
        raise ValueError(f"Invalid UCI move: {uci}")

    return (
        8 - int(uci[1]),
        ord(uci[0]) - ord("a"),
        8 - int(uci[3]),
        ord(uci[2]) - ord("a"),
    )


def move_to_action(move):
    r1, c1, r2, c2 = move

    return (r1 * 8 + c1) * 64 + (r2 * 8 + c2)


def evaluate_puzzle(network, encoder, row):
    game = ChessEnvironment()
    game.load_fen(row["FEN"])

    moves = row["Moves"].split()

    if not moves:
        raise ValueError("Puzzle has no solution moves")

    target_move = uci_to_move(moves[0])

    legal_moves = game.legal_moves()

    if target_move not in legal_moves:
        raise ValueError(
            f"Target move {moves[0]} is not legal"
        )

    legal_actions = [
        move_to_action(move)
        for move in legal_moves
    ]

    target_action = move_to_action(target_move)

    encoded = encoder.encode_flat(
        game.get_state()
    )

    result = network.forward(encoded)

    logits = result["policy"]

    # Rank only legal moves.
    ranked = sorted(
        legal_actions,
        key=lambda action: logits[action],
        reverse=True
    )

    rank = ranked.index(target_action) + 1

    # Stable masked softmax.
    maximum = max(
        logits[action]
        for action in legal_actions
    )

    values = {
        action: math.exp(
            logits[action] - maximum
        )
        for action in legal_actions
    }

    total = sum(values.values())

    target_probability = (
        values[target_action] / total
    )

    return rank, len(legal_actions), target_probability


def main():
    network = ORIONNetwork()
    encoder = ChessEncoder()
    checkpoint = Checkpoint(
        "checkpoints/v0.5_supervised"
    )

    episode = checkpoint.load(
        network,
        CHECKPOINT_PATH
    )

    with open(DATASET) as f:
        data = json.load(f)

    rows = data["rows"]

    top1 = 0
    top3 = 0
    top5 = 0

    total_rank = 0.0
    total_probability = 0.0

    valid = 0
    skipped = 0

    print("========================================")
    print("ORION v0.5.1 SUPERVISED EVALUATION")
    print("========================================")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Episode    : {episode}")
    print(f"Dataset    : {DATASET}")
    print(f"Puzzles    : {len(rows)}")
    print()

    for index, item in enumerate(rows, start=1):

        try:
            rank, legal_count, probability = (
                evaluate_puzzle(
                    network,
                    encoder,
                    item["row"]
                )
            )

            valid += 1
            total_rank += rank
            total_probability += probability

            if rank == 1:
                top1 += 1

            if rank <= 3:
                top3 += 1

            if rank <= 5:
                top5 += 1

            if index % 100 == 0:
                print(
                    f"Puzzle {index:04d}/1000 | "
                    f"valid={valid} | "
                    f"top1={top1 / valid:.4f} | "
                    f"avg_rank={total_rank / valid:.2f}"
                )

        except Exception as exc:
            skipped += 1

            print(
                f"[SKIP] Puzzle {index}: {exc}"
            )

    print()
    print("========================================")
    print("EVALUATION COMPLETE")
    print("========================================")

    print(f"Valid puzzles : {valid}")
    print(f"Skipped       : {skipped}")

    if valid == 0:
        print("[FAIL] No valid puzzles evaluated.")
        sys.exit(1)

    print()
    print(
        f"Top-1 accuracy : "
        f"{top1 / valid:.4f} "
        f"({top1}/{valid})"
    )

    print(
        f"Top-3 accuracy : "
        f"{top3 / valid:.4f} "
        f"({top3}/{valid})"
    )

    print(
        f"Top-5 accuracy : "
        f"{top5 / valid:.4f} "
        f"({top5}/{valid})"
    )

    print(
        f"Average rank   : "
        f"{total_rank / valid:.2f}"
    )

    print(
        f"Average target probability : "
        f"{total_probability / valid:.6f}"
    )

    print()
    print("[PASS] Supervised evaluation complete")


if __name__ == "__main__":
    main()
