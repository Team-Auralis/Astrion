import time

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork


game = ChessEnvironment()
encoder = ChessEncoder()
network = ORIONNetwork()

inputs = encoder.encode_flat(
    game.get_state()
)

runs = 10

start = time.perf_counter()

for _ in range(runs):
    network.forward(inputs)

elapsed = time.perf_counter() - start

# print("================================")
# print("       ORION BENCHMARK          ")
# print("================================")
# print()
# print("Runs          :", runs)
# print("Total seconds :", round(elapsed, 3))
# print(
#     "Per forward   :",
#     round(elapsed / runs, 3),
#     "seconds"
# )
# print(
#     "Forwards/sec  :",
#     round(runs / elapsed, 2)
# )
