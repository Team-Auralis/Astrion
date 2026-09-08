import math
import random


class ORIONNetwork:

    INPUT_SIZE = 896
    HIDDEN_1 = 256
    HIDDEN_2 = 128
    POLICY_SIZE = 4096

    def __init__(self, seed=42):

        random.seed(seed)

        self.w1 = self._matrix(
            self.HIDDEN_1,
            self.INPUT_SIZE
        )

        self.b1 = [0.0] * self.HIDDEN_1

        self.w2 = self._matrix(
            self.HIDDEN_2,
            self.HIDDEN_1
        )

        self.b2 = [0.0] * self.HIDDEN_2

        self.policy_w = self._matrix(
            self.POLICY_SIZE,
            self.HIDDEN_2
        )

        self.policy_b = [0.0] * self.POLICY_SIZE

        self.value_w = [
            self._random_weight()
            for _ in range(self.HIDDEN_2)
        ]

        self.value_b = 0.0

    def _random_weight(self):
        return random.uniform(-0.05, 0.05)

    def _matrix(self, rows, cols):

        return [
            [
                self._random_weight()
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

    @staticmethod
    def relu(x):
        return max(0.0, x)

    @staticmethod
    def sigmoid(x):

        if x < -60:
            return 0.0

        if x > 60:
            return 1.0

        return 1.0 / (1.0 + math.exp(-x))

    def dense(self, inputs, weights, bias):

        output = []

        for row, b in zip(weights, bias):

            total = b

            for x, w in zip(inputs, row):
                total += x * w

            output.append(total)

        return output

    def forward(self, inputs, return_hidden=False, return_cache=False):

        if len(inputs) != self.INPUT_SIZE:
            raise ValueError(
                f"Expected {self.INPUT_SIZE} inputs, "
                f"received {len(inputs)}"
            )

        # First hidden layer
        z1 = self.dense(
            inputs,
            self.w1,
            self.b1
        )

        h1 = [
            self.relu(x)
            for x in z1
        ]

        # Second hidden layer
        z2 = self.dense(
            h1,
            self.w2,
            self.b2
        )

        h2 = [
            self.relu(x)
            for x in z2
        ]

        # Policy
        policy = self.dense(
            h2,
            self.policy_w,
            self.policy_b
        )

        # Value
        value_raw = self.value_b

        for x, w in zip(h2, self.value_w):
            value_raw += x * w

        value = self.sigmoid(value_raw)

        result = {
            "policy": policy,
            "value": value
        }

        if return_hidden:
            result["hidden"] = h2

        if return_cache:
            result["cache"] = {
                "inputs": inputs,
                "z1": z1,
                "h1": h1,
                "z2": z2,
                "h2": h2,
                "policy": policy,
                "value_raw": value_raw,
                "value": value
            }

        return result

    def parameter_count(self):

        return (
            self.INPUT_SIZE * self.HIDDEN_1
            + self.HIDDEN_1
            + self.HIDDEN_1 * self.HIDDEN_2
            + self.HIDDEN_2
            + self.HIDDEN_2 * self.POLICY_SIZE
            + self.POLICY_SIZE
            + self.HIDDEN_2
            + 1
        )
