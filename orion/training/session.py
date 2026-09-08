from orion.training.train import ORIONTrainer
from orion.utils.checkpoint import Checkpoint


class TrainingSession:

    def __init__(
        self,
        seed=42,
        learning_rate=0.0001,
        temperature=1.0
    ):

        self.orion = ORIONTrainer(
            seed=seed,
            learning_rate=learning_rate,
            temperature=temperature
        )

        self.checkpoint = Checkpoint()

        self.wins = 0
        self.draws = 0
        self.losses = 0

    def run(
        self,
        episodes=10,
        max_moves=200
    ):

        results = []

        for episode in range(
            1,
            episodes + 1
        ):

            result = self.orion.train_game(
                max_moves=max_moves
            )

            winner = result["winner"]

            if winner == "white":

                self.wins += 1

            elif winner == "black":

                self.losses += 1

            else:

                self.draws += 1

            result["episode"] = episode

            results.append(result)

            print(
                f"Episode {episode:03d} | "
                f"winner={winner or 'draw':5s} | "
                f"moves={result['positions']:3d} | "
                f"policy={result['policy_loss']:.4f} | "
                f"value={result['value_loss']:.4f}"
            )

            self.checkpoint.save(
                self.orion.network,
                episode
            )

        return results
