from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.actions import ChessActionSpace
from orion.model.agent import ORIONAgent
from orion.training.reinforcement import ReinforcementTrainer


class ORIONTrainer:

    def __init__(
        self,
        seed=42,
        learning_rate=0.0001,
        temperature=1.0
    ):

        self.network = ORIONNetwork(
            seed=seed
        )

        self.encoder = ChessEncoder()

        self.agent = ORIONAgent(
            self.network,
            self.encoder,
            seed=seed
        )

        self.trainer = ReinforcementTrainer(
            self.network,
            learning_rate=learning_rate
        )

        self.temperature = temperature

    def train_game(self, max_moves=200):

        game = ChessEnvironment()

        records = []

        for _ in range(max_moves):

            status = game.game_status()

            if status.startswith("checkmate"):

                winner = status.split(":")[1]

                return self.update_game(
                    records,
                    winner
                )

            if status == "stalemate":

                return self.update_game(
                    records,
                    None
                )

            state = game.get_state()

            encoded = self.encoder.encode_flat(
                state
            )

            output = self.network.forward(
                encoded,
                return_cache=True
            )

            legal_moves = game.legal_moves()

            if not legal_moves:
                break

            legal_actions = [
                ChessActionSpace.encode(move)
                for move in legal_moves
            ]

            move = self.agent.choose_move(
                game,
                temperature=self.temperature
            )

            if move is None:
                break

            action = ChessActionSpace.encode(
                move
            )

            if not game.make_move(move):

                raise RuntimeError(
                    "ORION produced an illegal move."
                )

            # -------------------------------------------------
            # V0.5 TD VALUE TARGET
            #
            # The value head predicts the outcome for the
            # side to move.
            #
            # After this player moves, the opponent becomes
            # the side to move.
            #
            # Therefore:
            #
            # current_player_target =
            #     1.0 - opponent_next_state_value
            #
            # For a terminal position, the final game result
            # is handled later in update_game().
            # -------------------------------------------------

            next_status = game.game_status()

            next_value = None

            if (
                next_status == "playing"
                or next_status == "check"
            ):

                next_state = game.get_state()

                next_encoded = self.encoder.encode_flat(
                    next_state
                )

                next_output = self.network.forward(
                    next_encoded
                )

                next_value = next_output["value"]

            records.append({
                "cache": output["cache"],
                "action": action,
                "player": state["turn"],
                "legal_actions": legal_actions,
                "next_value": next_value
            })

        return self.update_game(
            records,
            None
        )

    def compute_value_target(
        self,
        player,
        winner,
        next_value
    ):
        # Non-terminal position:
        # next_value is from the opponent's perspective,
        # so invert it for the current player.
        if next_value is not None:
            return 1.0 - next_value

        # Terminal draw.
        if winner is None:
            return 0.5

        # Terminal win/loss.
        if player == winner:
            return 1.0

        return 0.0

    def update_game(
        self,
        records,
        winner
    ):

        policy_loss = 0.0
        value_loss = 0.0

        for record in records:

            player = record["player"]
            next_value = record["next_value"]

            # ------------------------------------------
            # V0.5 VALUE TARGET
            # ------------------------------------------
            #
            # If a next-state value exists, use one-step
            # TD-style propagation.
            #
            # The next state is from the opponent's
            # perspective, so invert it.
            #
            #     target = 1 - next_value
            #
            # If there is no next value, the position was
            # terminal and the actual game result is used.
            # ------------------------------------------

            value_target = self.compute_value_target(
                player=player,
                winner=winner,
                next_value=next_value
            )

            # ------------------------------------------
            # POLICY REWARD
            #
            # KEEP V0.4 POLICY OBJECTIVE UNCHANGED.
            # ------------------------------------------

            if winner is None:

                reward = 0.0

            elif player == winner:

                reward = 1.0

            else:

                reward = -1.0

            # ------------------------------------------
            # FULL BACKPROP
            # ------------------------------------------

            metrics = self.trainer.train_sample(
                cache=record["cache"],
                action=record["action"],
                reward=reward,
                value_target=value_target,
                legal_actions=record["legal_actions"]
            )

            policy_loss += metrics[
                "policy_loss"
            ]

            value_loss += metrics[
                "value_loss"
            ]

        return {
            "winner": winner,
            "positions": len(records),
            "policy_loss": policy_loss,
            "value_loss": value_loss
        }
