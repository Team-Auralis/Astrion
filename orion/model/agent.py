import math
import random

from orion.model.actions import ChessActionSpace


class ORIONAgent:

    def __init__(
        self,
        network,
        encoder,
        seed=None
    ):
        self.network = network
        self.encoder = encoder

        if seed is not None:
            random.seed(seed)

    def softmax(
        self,
        values,
        temperature=1.0
    ):

        if not values:
            return []

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than 0"
            )

        scaled = [
            value / temperature
            for value in values
        ]

        maximum = max(scaled)

        exponentials = [
            math.exp(value - maximum)
            for value in scaled
        ]

        total = sum(exponentials)

        if total <= 0:
            return [
                1.0 / len(values)
                for _ in values
            ]

        return [
            value / total
            for value in exponentials
        ]

    def sample_index(self, probabilities):

        if not probabilities:
            return None

        random_value = random.random()

        cumulative = 0.0

        for index, probability in enumerate(
            probabilities
        ):

            cumulative += probability

            if random_value <= cumulative:
                return index

        return len(probabilities) - 1

    def choose_move(
        self,
        environment,
        temperature=1.0
    ):

        state = environment.get_state()

        encoded = self.encoder.encode_flat(
            state
        )

        output = self.network.forward(
            encoded
        )

        policy = output["policy"]

        legal_moves = environment.legal_moves()

        if not legal_moves:
            return None

        legal_actions = [
            ChessActionSpace.encode(move)
            for move in legal_moves
        ]

        legal_scores = [
            policy[action]
            for action in legal_actions
        ]

        # Temperature <= 0 means deterministic play.
        if temperature <= 0:

            best_index = max(
                range(len(legal_scores)),
                key=lambda i: legal_scores[i]
            )

            return legal_moves[best_index]

        probabilities = self.softmax(
            legal_scores,
            temperature
        )

        selected_index = self.sample_index(
            probabilities
        )

        return legal_moves[selected_index]

    def analyze(self, environment):

        state = environment.get_state()

        encoded = self.encoder.encode_flat(
            state
        )

        output = self.network.forward(
            encoded
        )

        policy = output["policy"]

        legal_moves = environment.legal_moves()

        analysis = []

        for move in legal_moves:

            action = ChessActionSpace.encode(
                move
            )

            analysis.append({
                "move": move,
                "uci": ChessActionSpace.move_to_uci(
                    move
                ),
                "score": policy[action]
            })

        analysis.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "value": output["value"],
            "moves": analysis
        }
