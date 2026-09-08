from orion.model.network import ORIONNetwork
from orion.training.reinforcement import ReinforcementTrainer


network = ORIONNetwork(seed=42)
trainer = ReinforcementTrainer(
    network,
    learning_rate=0.0001
)

inputs = [0.0] * 896

# Give the network a few active features.
inputs[0] = 1.0
inputs[7] = 1.0
inputs[100] = 1.0

result = network.forward(
    inputs,
    return_cache=True
)

cache = result["cache"]

legal_actions = [
    100,
    200,
    300,
    400
]

action = 200

before_value = result["value"]

before_policy = result["policy"][action]

before_w1 = [
    row[:]
    for row in network.w1
]

metrics = trainer.train_sample(
    cache=cache,
    action=action,
    reward=1.0,
    value_target=1.0,
    legal_actions=legal_actions
)

after = network.forward(inputs)

after_value = after["value"]

after_policy = after["policy"][action]

after_w1 = [
    row[:]
    for row in network.w1
]

print("================================")
print("   ORION FULL BACKPROP TEST     ")
print("================================")

print("Policy loss       :", metrics["policy_loss"])
print("Value loss        :", metrics["value_loss"])

print()
print("Value before      :", before_value)
print("Value after       :", after_value)

print()
print("Selected logit before :", before_policy)
print("Selected logit after  :", after_policy)

w1_changes = 0

for before_row, after_row in zip(
    before_w1,
    after_w1
):
    for before, after in zip(
        before_row,
        after_row
    ):
        if before != after:
            w1_changes += 1

print()
print("W1 weights changed:", w1_changes)

assert metrics["policy_loss"] >= 0.0
assert metrics["value_loss"] >= 0.0

assert (
    after_value != before_value
)

assert (
    after_policy != before_policy
)

assert w1_changes > 0

print()
print("[PASS] Value head updated")
print("[PASS] Policy head updated")
print("[PASS] W1 weights updated")
print("[PASS] Full backpropagation reached input layer")

print()
print("ORION FULL BACKPROP ONLINE")
