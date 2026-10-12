from orion.environment.chess import ChessEnvironment
from orion.model.encoder import ChessEncoder
from orion.model.network import ORIONNetwork


game = ChessEnvironment()
encoder = ChessEncoder()
model = ORIONNetwork()

state = game.get_state()

inputs = encoder.encode_flat(state)

output = model.forward(inputs)

print("================================")
print("        ORION NEURAL NET        ")
print("================================")

print("Input size       :", len(inputs))
print("Policy size      :", len(output["policy"]))
print("Value            :", output["value"])
print("Parameters       :", model.parameter_count())

assert len(inputs) == 1152
assert len(output["policy"]) == 4096
assert 0.0 <= output["value"] <= 1.0

print()
print("[PASS] Forward pass")
print("[PASS] Policy head")
print("[PASS] Value head")
print()
print("ORION NETWORK ONLINE")
