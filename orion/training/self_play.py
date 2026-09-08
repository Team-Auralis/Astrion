from orion.model.agent import ORIONAgent


class SelfPlay:

    def __init__(self, environment_factory, agent):

        self.environment_factory = environment_factory
        self.agent = agent

    def play_game(self, max_moves=200):

        game = self.environment_factory()

        history = []

        for _ in range(max_moves):

            status = game.game_status()

            if status.startswith("checkmate"):
                winner = status.split(":")[1]

                return {
                    "history": history,
                    "winner": winner,
                    "moves": game.move_count,
                    "status": status
                }

            if status == "stalemate":
                return {
                    "history": history,
                    "winner": None,
                    "moves": game.move_count,
                    "status": status
                }

            state = game.get_state()

            move = self.agent.choose_move(
                game,
                temperature=1.0
            )

            if move is None:
                break

            history.append({
                "state": state,
                "move": move,
                "player": state["turn"]
            })

            success = game.make_move(move)

            if not success:
                raise RuntimeError(
                    "Agent produced an illegal move."
                )

        return {
            "history": history,
            "winner": None,
            "moves": game.move_count,
            "status": "move_limit"
        }
