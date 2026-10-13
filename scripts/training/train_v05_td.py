from pathlib import Path

from orion.training.train import ORIONTrainer
from orion.utils.checkpoint import Checkpoint


EPISODES = 5
MAX_MOVES = 200
LEARNING_RATE = 0.0001
TEMPERATURE = 1.0
SEED = 42

CHECKPOINT_DIR = Path("checkpoints/v0.5_td")


def main():

    checkpoint = Checkpoint(
        directory=CHECKPOINT_DIR
    )

    trainer = ORIONTrainer(
        seed=SEED,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE
    )

    total_wins = 0
    total_losses = 0
    total_draws = 0

    print("================================")
    print("    ORION v0.5 TD TRAINING")
    print("================================")
    print()
    print("Episodes       :", EPISODES)
    print("Max plies/game :", MAX_MOVES)
    print("Learning rate  :", LEARNING_RATE)
    print("Temperature    :", TEMPERATURE)
    print("Seed           :", SEED)
    print("Checkpoint dir :", CHECKPOINT_DIR)
    print()

    for episode in range(1, EPISODES + 1):

        result = trainer.train_game(
            max_moves=MAX_MOVES
        )

        winner = result["winner"]

        if winner == "white":
            total_wins += 1
        elif winner == "black":
            total_losses += 1
        else:
            total_draws += 1

        checkpoint_path = checkpoint.save(
            trainer.network,
            episode
        )

        print(
            f"Episode {episode:03d} | "
            f"winner={winner} | "
            f"positions={result['positions']} | "
            f"policy_loss={result['policy_loss']:.6f} | "
            f"value_loss={result['value_loss']:.6f} | "
            f"checkpoint={checkpoint_path}"
        )

    print()
    print("================================")
    print("      TRAINING COMPLETE")
    print("================================")
    print()
    print("White wins :", total_wins)
    print("Black wins :", total_losses)
    print("Draws      :", total_draws)
    print()
    print(
        "Final checkpoint:",
        CHECKPOINT_DIR / f"orion_{EPISODES}.json"
    )


if __name__ == "__main__":
    main()
