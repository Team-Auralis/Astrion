from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.agent import ORIONAgent
from orion.training.self_play import SelfPlay


network = ORIONNetwork()
encoder = ChessEncoder()
agent = ORIONAgent(network, encoder)

self_play = SelfPlay(
    ChessEnvironment,
    agent
)

print("================================")
print("       ORION SELF-PLAY v0.1     ")
print("================================")

game = self_play.play_game(
    max_moves=20
)

print()
print("Game status :", game["status"])
print("Winner      :", game["winner"])
print("Moves       :", game["moves"])
print("Positions   :", len(game["history"]))

assert len(game["history"]) <= 20

for position in game["history"][:3]:

    print(
        position["player"],
        "->",
        ChessEnvironment().move_to_text(
            position["move"]
        )
    )

print()
print("[PASS] Self-play completed")
print("[PASS] Game history recorded")
print()
print("SELF-PLAY ENGINE ONLINE")
