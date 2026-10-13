import json
import math
import random
from pathlib import Path

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.actions import ChessActionSpace


DATASET_FILE = Path("lichess_sample.json")
CHECKPOINT_DIR = Path("checkpoints/v0.5_supervised")

PUZZLES = 1000
LEARNING_RATE = 0.0001
SEED = 42


def load_puzzles(path):
    with open(path) as file:
        data = json.load(file)

    return [
        row["row"]
        for row in data["rows"]
    ]


def uci_to_move(uci):
    if len(uci) != 4:
        raise ValueError(f"Invalid UCI move: {uci}")

    return (
        8 - int(uci[1]),
        ord(uci[0]) - ord("a"),
        8 - int(uci[3]),
        ord(uci[2]) - ord("a")
    )


def softmax(logits):
    maximum = max(logits)

    values = [
        math.exp(x - maximum)
        for x in logits
    ]

    total = sum(values)

    return [
        x / total
        for x in values
    ]


def masked_softmax(logits, legal_actions):
    values = [
        logits[action]
        for action in legal_actions
    ]

    probabilities = softmax(values)

    return {
        action: probability
        for action, probability
        in zip(legal_actions, probabilities)
    }


def validate_puzzle(puzzle, encoder):
    game = ChessEnvironment()

    game.load_fen(puzzle["FEN"])

    moves = puzzle["Moves"].split()

    if not moves:
        return None

    target_move = uci_to_move(moves[0])

    legal_moves = game.legal_moves()

    if target_move not in legal_moves:
        return None

    encoded = encoder.encode_flat(
        game.get_state()
    )

    target_action = ChessActionSpace.encode(
        target_move
    )

    legal_actions = [
        ChessActionSpace.encode(move)
        for move in legal_moves
    ]

    return (
        encoded,
        target_action,
        legal_actions
    )


def train_sample(
    network,
    encoded,
    target_action,
    legal_actions,
    learning_rate
):

    output = network.forward(
        encoded,
        return_cache=True
    )

    logits = output["policy"]

    probabilities = masked_softmax(
        logits,
        legal_actions
    )

    target_probability = max(
        probabilities[target_action],
        1e-12
    )

    loss = -math.log(
        target_probability
    )

    gradient = {
        action: probability
        for action, probability in probabilities.items()
    }

    gradient[target_action] -= 1.0

    cache = output["cache"]

    # Policy-head gradient
    h2 = cache["h2"]

    policy_w = network.policy_w

    d_policy_w = [
        [
            gradient[action] * h2[j]
            for j in range(len(h2))
        ]
        for action in range(network.POLICY_SIZE)
    ]

    d_policy_b = [
        gradient[action]
        for action in range(network.POLICY_SIZE)
    ]

    # Backprop into hidden layer.
    d_h2 = [
        sum(
            gradient[action] * policy_w[action][j]
            for action in range(network.POLICY_SIZE)
        )
        for j in range(len(h2))
    ]

    d_z2 = [
        value if cache["z2"][j] > 0 else 0.0
        for j, value in enumerate(d_h2)
    ]

    h1 = cache["h1"]

    d_w2 = [
        [
            d_z2[j] * h1[i]
            for i in range(len(h1))
        ]
        for j in range(network.HIDDEN_2)
    ]

    d_b2 = d_z2

    d_h1 = [
        sum(
            d_z2[j] * network.w2[j][i]
            for j in range(network.HIDDEN_2)
        )
        for i in range(network.HIDDEN_1)
    ]

    d_z1 = [
        value if cache["z1"][i] > 0 else 0.0
        for i, value in enumerate(d_h1)
    ]

    inputs = cache["inputs"]

    d_w1 = [
        [
            d_z1[j] * inputs[i]
            for i in range(network.INPUT_SIZE)
        ]
        for j in range(network.HIDDEN_1)
    ]

    d_b1 = d_z1

    # Gradient descent.
    for action in range(network.POLICY_SIZE):

        for j in range(network.HIDDEN_2):

            network.policy_w[action][j] -= (
                learning_rate *
                d_policy_w[action][j]
            )

        network.policy_b[action] -= (
            learning_rate *
            d_policy_b[action]
        )

    for j in range(network.HIDDEN_2):

        for i in range(network.HIDDEN_1):

            network.w2[j][i] -= (
                learning_rate *
                d_w2[j][i]
            )

        network.b2[j] -= (
            learning_rate *
            d_b2[j]
        )

    for j in range(network.HIDDEN_1):

        for i in range(network.INPUT_SIZE):

            network.w1[j][i] -= (
                learning_rate *
                d_w1[j][i]
            )

        network.b1[j] -= (
            learning_rate *
            d_b1[j]
        )

    return loss


def main():

    random.seed(SEED)

    print("================================")
    print(" ORION v0.5 SUPERVISED TRAINING")
    print("================================")
    print()

    print("Dataset:", DATASET_FILE)
    print("Requested puzzles:", PUZZLES)
    print("Learning rate:", LEARNING_RATE)
    print()

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}"
        )

    puzzles = load_puzzles(
        DATASET_FILE
    )

    if len(puzzles) < PUZZLES:
        print(
            f"Only {len(puzzles)} puzzles available."
        )
        print(
            "This is expected for the 5-row sample."
        )
        print(
            "Training will use the available rows."
        )

    encoder = ChessEncoder()
    network = ORIONNetwork(
        seed=SEED
    )

    valid = 0
    invalid = 0
    total_loss = 0.0
    total = min(
        PUZZLES,
        len(puzzles)
    )

    for index in range(total):

        puzzle = puzzles[index]

        try:

            sample = validate_puzzle(
                puzzle,
                encoder
            )

        except Exception:

            sample = None

        if sample is None:

            invalid += 1
            continue

        encoded, target_action, legal_actions = sample

        loss = train_sample(
            network,
            encoded,
            target_action,
            legal_actions,
            LEARNING_RATE
        )

        valid += 1
        total_loss += loss

        print(
            f"Puzzle {index + 1:04d} | "
            f"loss={loss:.6f}"
        )

    print()
    print("================================")
    print("       TRAINING SUMMARY")
    print("================================")
    print()
    print("Processed :", total)
    print("Valid     :", valid)
    print("Invalid   :", invalid)

    if valid:
        print(
            "Avg loss  :",
            total_loss / valid
        )

    print()
    print("Supervised smoke test complete.")


if __name__ == "__main__":
    main()
