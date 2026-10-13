import time

from orion.environment.chess import ChessEnvironment


game = ChessEnvironment()

runs = 1000

start = time.perf_counter()

for _ in range(runs):
    game.legal_moves()

elapsed = time.perf_counter() - start

print("================================")
print("   ORION ENVIRONMENT BENCHMARK  ")
print("================================")

print("Runs          :", runs)
print("Total seconds :", round(elapsed, 3))
print(
    "Per legal-move:",
    round(elapsed / runs * 1000, 3),
    "ms"
)
print(
    "Calls/sec     :",
    round(runs / elapsed, 2)
)
