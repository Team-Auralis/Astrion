from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.actions import ChessActionSpace
from orion.training.policy_trainer import PolicyTrainer


game = ChessEnvironment()
encoder = ChessEncoder()
network = ORIONNetwork()
trainer = PolicyTrainer(
    network,
    learning_rate=0.01
)

state = game.get_state()
inputs = encoder.encode_flat(state)

target_move = (6, 4, 4, 4)  # e2-e4
target_action = ChessActionSpace.encode(target_move)

before = network.forward(
    inputs,
    return_hidden=True
)

loss_before = trainer.train_sample(
    before["hidden"],
    before["policy"],
    target_action
)

for _ in range(20):

    output = network.forward(
        inputs,
        return_hidden=True
    )

    trainer.train_sample(
        output["hidden"],
        output["policy"],
        target_action
    )

after = network.forward(
    inputs,
    return_hidden=True
)

# Compute final target probability.
trainer_probs = trainer.softmax(
    after["policy"]
)

target_probability = trainer_probs[target_action]

loss_after = -__import__("math").log(
    max(target_probability, 1e-12)
)

print("================================")
print("      ORION POLICY LEARNING     ")
print("================================")

print("Target move       :", "e2e4")
print("Target action     :", target_action)
print("Initial loss      :", loss_before)
print("Final loss        :", loss_after)
print("Final probability :", target_probability)

assert loss_after < loss_before

print()
print("[PASS] Policy loss decreased")
print("[PASS] Policy head updated")
print()
print("ORION POLICY LEARNING WORKS")
