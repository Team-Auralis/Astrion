from collections import Counter

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.agent import ORIONAgent


network = ORIONNetwork(seed=42)
encoder = ChessEncoder()

agent = ORIONAgent(
    network,
    encoder,
    seed=123
)

counts = Counter()

runs = 100

for _ in range(runs):

    game = ChessEnvironment()

    move = agent.choose_move(
        game,
        temperature=1.0
    )

    counts[move] += 1


print("================================")
print("     ORION EXPLORATION TEST     ")
print("================================")

print("Samples:", runs)
print("Unique moves:", len(counts))

for move, count in counts.most_common():
    print(
        f"{move}: {count}"
    )

assert len(counts) > 1

print()
print("[PASS] ORION samples multiple legal moves")
print("ORION EXPLORATION ONLINE")
