import math

from orion.model.network import ORIONNetwork
from orion.training.reinforcement import ReinforcementTrainer


network = ORIONNetwork(seed=42)

trainer = ReinforcementTrainer(
    network,
    learning_rate=0.0001
)

inputs = [0.0] * 896

# Activate several features.
inputs[0] = 1.0
inputs[7] = 1.0
inputs[100] = 1.0
inputs[200] = 1.0

legal_actions = [
    100,
    200,
    300,
    400
]

target_action = 200


def probability(logits, action):
    maximum = max(
        logits[a]
        for a in legal_actions
    )

    values = {
        a: math.exp(logits[a] - maximum)
        for a in legal_actions
    }

    total = sum(values.values())

    return values[action] / total


before = network.forward(
    inputs,
    return_cache=True
)

before_probability = probability(
    before["policy"],
    target_action
)

print("================================")
print(" ORION LEARNING DIRECTION TEST ")
print("================================")

print(
    "Initial probability :",
    before_probability
)

steps = 20

for step in range(steps):

    result = network.forward(
        inputs,
        return_cache=True
    )

    trainer.train_sample(
        cache=result["cache"],
        action=target_action,
        reward=1.0,
        value_target=1.0,
        legal_actions=legal_actions
    )

after = network.forward(inputs)

after_probability = probability(
    after["policy"],
    target_action
)

print(
    "Final probability   :",
    after_probability
)

print(
    "Probability change  :",
    after_probability -
    before_probability
)

print()

if after_probability > before_probability:

    print(
        "[PASS] Target action probability increased"
    )

else:

    print(
        "[FAIL] Target action probability did not increase"
    )

assert after_probability > before_probability

print()
print("ORION LEARNING DIRECTION VERIFIED")
