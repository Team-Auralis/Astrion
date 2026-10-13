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

        if status != "playing" and status != "check":
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

        success = game.make_move(move)

        if not success:
            raise RuntimeError(
                f"Illegal move returned by agent: {move}"
            )

    status = game.game_status()

    return {
        "status": status,
        "winner": (
            status.split(":")[1]
            if status.startswith("checkmate:")
            else None
        ),
        "moves": game.move_count
    }


def evaluate(
    games=10,
    orion_color="white",
    seed=1000,
    checkpoint_path=None
):

    network = ORIONNetwork()

    loaded_episode = None

    if checkpoint_path is not None:

        checkpoint = Checkpoint(
            directory=Path(checkpoint_path).parent
        )

        loaded_episode = checkpoint.load(
            network,
            checkpoint_path
        )

    encoder = ChessEncoder()

    orion = ORIONAgent(
        network,
        encoder,
        seed=seed
    )

    results = []

    for i in range(games):

        result = play_game(
            orion,
            orion_color,
            seed + i
        )

        results.append(result)

    wins = 0
    losses = 0
    draws = 0

    reasons = Counter()
    total_moves = 0

    for result in results:

        status = result["status"]

        total_moves += result["moves"]

        if status.startswith("checkmate:"):

            winner = result["winner"]

            if winner == orion_color:
                wins += 1
            else:
                losses += 1

            reasons["checkmate"] += 1

        else:

            draws += 1
            reasons[status] += 1

    print("================================")
    print("      ORION v0.6 EVALUATION")
    print("================================")
    print()

    if checkpoint_path is None:
        print("Network        : fresh initialization")
    else:
        print("Network        : trained checkpoint")
        print("Checkpoint     :", checkpoint_path)
        print("Episode        :", loaded_episode)

    print("Opponent       : fixed random")
    print("ORION color    :", orion_color)
    print("Games          :", games)
    print("Wins           :", wins)
    print("Losses         :", losses)
    print("Draws          :", draws)

    print(
        "Win rate       :",
        round(wins / games, 3)
    )

    print(
        "Loss rate      :",
        round(losses / games, 3)
    )

    print(
        "Draw rate      :",
        round(draws / games, 3)
    )

    print(
        "Average plies  :",
        round(total_moves / games, 2)
    )

    print()
    print("Termination reasons:")

    for reason, count in reasons.items():
        print(f"  {reason}: {count}")

    print()

    score = (
        wins + 0.5 * draws
    ) / games

    print(
        "Chess score     :",
        round(score, 3)
    )

    print()
    print("ORION v0.6 EVALUATION COMPLETE")


if __name__ == "__main__":

    evaluate(
        games=10,
        orion_color="white",
        seed=1000
    )
