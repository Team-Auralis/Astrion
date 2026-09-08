import json
import os
import sys

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.training.reinforcement import ReinforcementTrainer
from orion.utils.checkpoint import Checkpoint


DATASET = "lichess_1000.json"
CHECKPOINT_DIR = "checkpoints/v0.5_supervised"


def uci_to_move(uci):
    if len(uci) != 4:
        raise ValueError(f"Invalid UCI move: {uci}")

    return (
        8 - int(uci[1]),
        ord(uci[0]) - ord("a"),
        8 - int(uci[3]),
        ord(uci[2]) - ord("a"),
    )


def move_to_action(move):
    r1, c1, r2, c2 = move

    from_square = r1 * 8 + c1
    to_square = r2 * 8 + c2

    return from_square * 64 + to_square


def validate_puzzle(row):
    game = ChessEnvironment()
    game.load_fen(row["FEN"])

    moves = row["Moves"].split()

    if not moves:
        raise ValueError("Puzzle has no moves")

    target_uci = moves[0]
    target_move = uci_to_move(target_uci)

    legal_moves = game.legal_moves()

    if target_move not in legal_moves:
        raise ValueError(
            f"Target move {target_uci} is not legal"
        )

    target_action = move_to_action(target_move)

    return game, target_move, target_action, legal_moves


def main():
    if not os.path.exists(DATASET):
        print(f"[ERROR] Dataset not found: {DATASET}")
        sys.exit(1)

    with open(DATASET) as f:
        data = json.load(f)

    rows = data["rows"]

    print("========================================")
    print("ORION v0.5 SUPERVISED TRAINING")
    print("========================================")
    print(f"Dataset : {DATASET}")
    print(f"Puzzles : {len(rows)}")
    print()

    encoder = ChessEncoder()
    network = ORIONNetwork()
    trainer = ReinforcementTrainer(network)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    total_loss = 0.0
    valid = 0
    skipped = 0

    for index, item in enumerate(rows, start=1):
        row = item["row"]

        try:
            game, target_move, target_action, legal_moves = validate_puzzle(row)

            state = game.get_state()
            encoded = encoder.encode_flat(state)

            result = network.forward(
                encoded,
                return_cache=True
            )

            cache = result["cache"]

            loss = trainer.train_supervised_sample(
                cache=cache,
                target_action=target_action,
                legal_actions=[
                    move_to_action(move)
                    for move in legal_moves
                ],
            )

            total_loss += loss["policy_loss"]
            valid += 1

            if index % 100 == 0:
                avg_loss = total_loss / valid

                print(
                    f"Puzzle {index:04d}/1000 | "
                    f"valid={valid} | "
                    f"avg_policy_loss={avg_loss:.6f}"
                )

        except Exception as exc:
            skipped += 1

            print(
                f"[SKIP] Puzzle {index}: {exc}"
            )

    print()
    print("========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(f"Valid puzzles : {valid}")
    print(f"Skipped       : {skipped}")

    if valid:
        print(
            f"Average policy loss : "
            f"{total_loss / valid:.6f}"
        )

    if valid == 0:
        print()
        print("[FAIL] No puzzles were successfully trained.")
        sys.exit(1)

    checkpoint = Checkpoint(CHECKPOINT_DIR)

    checkpoint_path = checkpoint.save(
        network,
        episode=1000
    )

    print()
    print(f"Checkpoint saved: {checkpoint_path}")

    print()
    print("[PASS] ORION v0.5 supervised training complete")


if __name__ == "__main__":
    main()
