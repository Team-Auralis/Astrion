import json
import os
import random
import sys

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.training.reinforcement import ReinforcementTrainer
from orion.utils.checkpoint import Checkpoint


DATASET = "lichess_1000.json"
CHECKPOINT_DIR = "checkpoints/v0.5.2_lr0005_supervised"

EPOCHS = 5
LEARNING_RATE = 0.0005
SEED = 42


def uci_to_move(uci):
    """
    Convert UCI coordinates to ORION's internal move format.

    Supports normal moves:
        e2e4

    And promotion moves:
        e7e8q
        e7e8r
        e7e8b
        e7e8n

    The current 4096-action space represents source/destination
    squares only, so the promotion piece is intentionally ignored.
    """

    if len(uci) not in (4, 5):
        raise ValueError(f"Invalid UCI move: {uci}")

    if (
        uci[0] not in "abcdefgh"
        or uci[2] not in "abcdefgh"
        or uci[1] not in "12345678"
        or uci[3] not in "12345678"
    ):
        raise ValueError(f"Invalid UCI move: {uci}")

    if len(uci) == 5 and uci[4].lower() not in "qrbn":
        raise ValueError(f"Invalid promotion move: {uci}")

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


def prepare_puzzle(row):
    game = ChessEnvironment()
    game.load_fen(row["FEN"])

    moves = row["Moves"].split()

    if not moves:
        raise ValueError("Puzzle has no solution moves")

    target_uci = moves[0]
    target_move = uci_to_move(target_uci)

    legal_moves = game.legal_moves()

    if target_move not in legal_moves:
        raise ValueError(
            f"Target move {target_uci} is not legal"
        )

    legal_actions = [
        move_to_action(move)
        for move in legal_moves
    ]

    target_action = move_to_action(target_move)

    return game, target_action, legal_actions


def main():

    random.seed(SEED)

    if not os.path.exists(DATASET):
        print(f"[ERROR] Dataset not found: {DATASET}")
        sys.exit(1)

    with open(DATASET) as f:
        data = json.load(f)

    rows = data["rows"]

    print("========================================")
    print("ORION v0.5.1 SUPERVISED TRAINING")
    print("========================================")
    print(f"Dataset        : {DATASET}")
    print(f"Puzzles        : {len(rows)}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Random seed    : {SEED}")
    print()

    # Prepare and validate all puzzles once.
    prepared = []
    skipped = 0

    print("Preparing dataset...")

    for index, item in enumerate(rows, start=1):

        try:
            game, target_action, legal_actions = (
                prepare_puzzle(item["row"])
            )

            prepared.append(
                (
                    game,
                    target_action,
                    legal_actions
                )
            )

        except Exception as exc:
            skipped += 1
            print(
                f"[SKIP] Puzzle {index}: {exc}"
            )

    print()
    print(f"Prepared puzzles : {len(prepared)}")
    print(f"Skipped          : {skipped}")

    if not prepared:
        print("[FAIL] No valid training samples.")
        sys.exit(1)

    encoder = ChessEncoder()
    network = ORIONNetwork()

    trainer = ReinforcementTrainer(
        network,
        learning_rate=LEARNING_RATE
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print()
    print("Starting training...")
    print()

    for epoch in range(1, EPOCHS + 1):

        order = list(range(len(prepared)))
        random.shuffle(order)

        epoch_loss = 0.0
        epoch_valid = 0

        for position in order:

            game, target_action, legal_actions = (
                prepared[position]
            )

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
                legal_actions=legal_actions
            )

            epoch_loss += loss["policy_loss"]
            epoch_valid += 1

        average_loss = epoch_loss / epoch_valid

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"trained={epoch_valid} | "
            f"avg_policy_loss={average_loss:.6f}"
        )

    print()
    print("========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(f"Valid puzzles : {len(prepared)}")
    print(f"Skipped       : {skipped}")
    print(f"Epochs        : {EPOCHS}")

    checkpoint = Checkpoint(CHECKPOINT_DIR)

    checkpoint_path = checkpoint.save(
        network,
        episode=EPOCHS
    )

    print()
    print(f"Checkpoint saved: {checkpoint_path}")
    print()
    print("[PASS] ORION v0.5.1 supervised training complete")


if __name__ == "__main__":
    main()
