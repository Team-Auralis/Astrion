import json
import math
import random

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.training.reinforcement import ReinforcementTrainer


DATASET = "lichess_1000.json"

SAMPLES = 20
EPOCHS = 100
LEARNING_RATE = 0.0001
SEED = 42


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


def prepare_puzzles(rows):
    encoder = ChessEncoder()
    prepared = []

    for index, item in enumerate(rows[:SAMPLES], start=1):
        game = ChessEnvironment()
        game.load_fen(item["row"]["FEN"])

        moves = item["row"]["Moves"].split()

        if not moves:
            raise ValueError(
                f"Puzzle {index} has no solution moves"
            )

        target_move = uci_to_move(moves[0])
        legal_moves = game.legal_moves()

        if target_move not in legal_moves:
            raise ValueError(
                f"Puzzle {index}: target move is illegal"
            )

        encoded = encoder.encode_flat(
            game.get_state()
        )

        legal_actions = [
            move_to_action(move)
            for move in legal_moves
        ]

        target_action = move_to_action(
            target_move
        )

        prepared.append({
            "encoded": encoded,
            "legal_actions": legal_actions,
            "target_action": target_action,
        })

    return prepared


def evaluate(network, samples):
    total_loss = 0.0
    top1 = 0
    top3 = 0
    top5 = 0

    for sample in samples:
        result = network.forward(
            sample["encoded"]
        )

        logits = result["policy"]
        legal_actions = sample["legal_actions"]
        target = sample["target_action"]

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

        probability = values[target] / total

        loss = -math.log(
            max(probability, 1e-12)
        )

        total_loss += loss

        ranked = sorted(
            legal_actions,
            key=lambda action: logits[action],
            reverse=True
        )

        rank = ranked.index(target) + 1

        if rank == 1:
            top1 += 1

        if rank <= 3:
            top3 += 1

        if rank <= 5:
            top5 += 1

    count = len(samples)

    return {
        "loss": total_loss / count,
        "top1": top1 / count,
        "top3": top3 / count,
        "top5": top5 / count,
    }


def main():
    print("========================================")
    print("ORION SUPERVISED OVERFIT DIAGNOSTIC")
    print("========================================")
    print(f"Dataset        : {DATASET}")
    print(f"Samples        : {SAMPLES}")
    print(f"Epochs         : {EPOCHS}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Random seed    : {SEED}")
    print()

    random.seed(SEED)

    with open(DATASET) as file:
        data = json.load(file)

    samples = prepare_puzzles(
        data["rows"]
    )

    print(
        f"Prepared samples : {len(samples)}"
    )
    print()

    network = ORIONNetwork(
        seed=SEED
    )

    trainer = ReinforcementTrainer(
        network,
        learning_rate=LEARNING_RATE
    )

    initial = evaluate(
        network,
        samples
    )

    print("Initial evaluation")
    print(
        f"Loss  : {initial['loss']:.6f}"
    )
    print(
        f"Top-1 : {initial['top1']:.4f}"
    )
    print(
        f"Top-3 : {initial['top3']:.4f}"
    )
    print(
        f"Top-5 : {initial['top5']:.4f}"
    )
    print()

    for epoch in range(1, EPOCHS + 1):

        order = list(
            range(len(samples))
        )

        random.shuffle(order)

        for index in order:

            sample = samples[index]

            result = network.forward(
                sample["encoded"],
                return_cache=True
            )

            trainer.train_supervised_sample(
                cache=result["cache"],
                target_action=sample["target_action"],
                legal_actions=sample["legal_actions"]
            )

        metrics = evaluate(
            network,
            samples
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"loss={metrics['loss']:.6f} | "
                f"top1={metrics['top1']:.4f} | "
                f"top3={metrics['top3']:.4f} | "
                f"top5={metrics['top5']:.4f}"
            )

    final = evaluate(
        network,
        samples
    )

    print()
    print("========================================")
    print("OVERFIT DIAGNOSTIC COMPLETE")
    print("========================================")

    print(
        f"Initial loss : {initial['loss']:.6f}"
    )
    print(
        f"Final loss   : {final['loss']:.6f}"
    )

    print(
        f"Initial Top-1 : {initial['top1']:.4f}"
    )
    print(
        f"Final Top-1   : {final['top1']:.4f}"
    )

    print(
        f"Initial Top-3 : {initial['top3']:.4f}"
    )
    print(
        f"Final Top-3   : {final['top3']:.4f}"
    )

    print(
        f"Initial Top-5 : {initial['top5']:.4f}"
    )
    print(
        f"Final Top-5   : {final['top5']:.4f}"
    )

    if final["loss"] < initial["loss"]:
        print()
        print(
            "[PASS] Training loss decreased"
        )
    else:
        print()
        print(
            "[FAIL] Training loss did not decrease"
        )


if __name__ == "__main__":
    main()
