from orion.training.train import ORIONTrainer


trainer = ORIONTrainer(
    seed=123,
    learning_rate=0.0001,
    temperature=1.0
)

game = trainer.orion if hasattr(trainer, "orion") else None

# Create a real chess position.
from orion.environment.chess import ChessEnvironment

environment = ChessEnvironment()

state = environment.get_state()
encoded = trainer.encoder.encode_flat(state)

output = trainer.network.forward(
    encoded,
    return_cache=True
)

legal_moves = environment.legal_moves()

from orion.model.actions import ChessActionSpace

legal_actions = [
    ChessActionSpace.encode(move)
    for move in legal_moves
]

action = legal_actions[0]

before = output["value"]

target = trainer.compute_value_target(
    player=state["turn"],
    winner=None,
    next_value=0.20
)

metrics = trainer.trainer.train_sample(
    cache=output["cache"],
    action=action,
    reward=0.0,
    value_target=target,
    legal_actions=legal_actions
)

after = trainer.network.forward(encoded)["value"]

print("================================")
print("     ORION v0.5 TD LEARNING")
print("================================")
print()
print("Initial value :", before)
print("TD target     :", target)
print("Final value   :", after)
print("Value loss    :", metrics["value_loss"])
print()

assert metrics["value_loss"] > 0.0
assert abs(after - target) < abs(before - target)

print("[PASS] TD value loss generated")
print("[PASS] Value moved toward TD target")
print()
print("ORION v0.5 TD LEARNING VERIFIED")
