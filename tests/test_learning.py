from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.training.trainer import ORIONTrainer


game = ChessEnvironment()
encoder = ChessEncoder()
network = ORIONNetwork()
trainer = ORIONTrainer(
    network,
    learning_rate=0.05
)

state = game.get_state()
inputs = encoder.encode_flat(state)

before = network.forward(
    inputs,
    return_hidden=True
)

target = 1.0

loss_before = trainer.value_loss(
    before["value"],
    target
)

for _ in range(100):

    output = network.forward(
        inputs,
        return_hidden=True
    )

    trainer.train_value_sample(
        output["hidden"],
        output["value"],
        target
    )

after = network.forward(
    inputs,
    return_hidden=True
)

loss_after = trainer.value_loss(
    after["value"],
    target
)

print("================================")
print("       ORION LEARNING TEST       ")
print("================================")

print("Initial value :", before["value"])
print("Final value   :", after["value"])
print("Initial loss  :", loss_before)
print("Final loss    :", loss_after)

assert loss_after < loss_before

print()
print("[PASS] Value prediction improved")
print("[PASS] Gradient update worked")
print()
print("ORION CAN LEARN")
