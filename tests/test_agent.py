from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.agent import ORIONAgent


game = ChessEnvironment()
encoder = ChessEncoder()
network = ORIONNetwork()
agent = ORIONAgent(network, encoder)

print("================================")
print("        ORION AGENT v0.1        ")
print("================================")

game.display()

move = agent.choose_move(game)

print("ORION selected:", game.move_to_text(move))

assert move in game.legal_moves()

success = game.make_move(move)

assert success

print()
print("[PASS] ORION selected a legal move")
print("[PASS] Move executed successfully")

print()
game.display()

analysis = agent.analyze(game)

print("Position value:", analysis["value"])

print()
print("Top candidate moves:")

for item in analysis["moves"][:5]:
    print(
        f"  {item['uci']} "
        f"score={item['score']:.6f}"
    )

print()
print("ORION AGENT ONLINE")
