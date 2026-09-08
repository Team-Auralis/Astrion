import time

from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork


game = ChessEnvironment()
encoder = ChessEncoder()
network = ORIONNetwork()

x = encoder.encode_flat(game.get_state())

runs = 20


def benchmark(name, fn):
    start = time.perf_counter()

    for _ in range(runs):
        fn()

    elapsed = time.perf_counter() - start

    print(
        f"{name:<20}"
        f"{elapsed / runs * 1000:.3f} ms"
    )


print("================================")
print("       ORION LAYER BENCHMARK    ")
print("================================")
print("Runs:", runs)
print()

benchmark(
    "Layer 1",
    lambda: network.dense(
        x,
        network.w1,
        network.b1
    )
)

h1 = network.dense(x, network.w1, network.b1)
h1 = [network.relu(v) for v in h1]

benchmark(
    "Layer 2",
    lambda: network.dense(
        h1,
        network.w2,
        network.b2
    )
)

h2 = network.dense(h1, network.w2, network.b2)
h2 = [network.relu(v) for v in h2]

benchmark(
    "Policy head",
    lambda: network.dense(
        h2,
        network.policy_w,
        network.policy_b
    )
)

benchmark(
    "Value head",
    lambda: network.sigmoid(
        network.value_b
        + sum(
            v * w
            for v, w in zip(h2, network.value_w)
        )
    )
)

benchmark(
    "Full forward",
    lambda: network.forward(x)
)
