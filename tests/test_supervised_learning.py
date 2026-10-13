from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork
from orion.model.actions import ChessActionSpace
from orion.training.reinforcement import ReinforcementTrainer


game = ChessEnvironment()

encoder = ChessEncoder()
network = ORIONNetwork(seed=42)

trainer = ReinforcementTrainer(
    network,
    learning_rate=0.0001
)

state = game.get_state()

encoded = encoder.encode_flat(state)

output = network.forward(
    encoded,
    return_cache=True
)

legal_moves = game.legal_moves()

legal_actions = [
    ChessActionSpace.encode(move)
    for move in legal_moves
]

target_action = legal_actions[0]

before = network.forward(encoded)

metrics = trainer.train_supervised_sample(
    cache=output["cache"],
    target_action=target_action,
    legal_actions=legal_actions
)

after = network.forward(encoded)

print("================================")
print(" ORION SUPERVISED LEARNING TEST")
print("================================")
print()
print("Target action :", target_action)
print("Policy loss   :", metrics["policy_loss"])
print("Value loss    :", metrics["value_loss"])
print("Value before  :", before["value"])
print("Value after   :", after["value"])
print()

assert metrics["policy_loss"] > 0.0

assert metrics["value_loss"] == 0.0

assert after["policy"][target_action] != before["policy"][target_action]

print("[PASS] Supervised policy loss generated")
print("[PASS] Policy parameters updated")
print("[PASS] Value loss is zero")
print("[PASS] Value head is excluded from supervised target")
print()
print("ORION SUPERVISED LEARNING VERIFIED")
