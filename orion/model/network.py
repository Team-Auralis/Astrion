import math
import random


class ResidualBlock:
    """Single residual block with two FC layers (BN fused at export)."""

    def __init__(self, channels, rng):
        self.w_a = _matrix(rng, channels, channels)
        self.b_a = [0.0] * channels
        self.w_b = _matrix(rng, channels, channels)
        self.b_b = [0.0] * channels


def _rng_weight(rng):
    limit = 1.0 / math.sqrt(256)
    return rng.uniform(-limit, limit)


def _matrix(rng, rows, cols):
    limit = 1.0 / math.sqrt(cols)
    return [[rng.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


class ORIONNetwork:

    INPUT_SIZE = 1152
    CHANNELS = 256
    NUM_BLOCKS = 6
    POLICY_SIZE = 4096

    def __init__(self, seed=42):

        rng = random.Random(seed)

        self.input_w = _matrix(rng, self.CHANNELS, self.INPUT_SIZE)
        self.input_b = [0.0] * self.CHANNELS

        self.blocks = [ResidualBlock(self.CHANNELS, rng) for _ in range(self.NUM_BLOCKS)]

        self.policy_w1 = _matrix(rng, 512, self.CHANNELS)
        self.policy_b1 = [0.0] * 512
        self.policy_w2 = _matrix(rng, self.POLICY_SIZE, 512)
        self.policy_b2 = [0.0] * self.POLICY_SIZE

        self.value_w1 = _matrix(rng, 64, self.CHANNELS)
        self.value_b1 = [0.0] * 64
        self.value_w2 = [rng.uniform(-0.01, 0.01) for _ in range(64)]
        self.value_b2 = 0.0

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

        h = self.dense(inputs, self.input_w, self.input_b)
        h = [self.relu(x) for x in h]

        for block in self.blocks:
            residual = h[:]
            z = self.dense(h, block.w_a, block.b_a)
            z = [self.relu(x) for x in z]
            z = self.dense(z, block.w_b, block.b_b)
            h = [r + z_i for r, z_i in zip(residual, z)]
            h = [self.relu(x) for x in h]

        policy_logits = self.dense(h, self.policy_w2, self.policy_b2)

        v = self.dense(h, self.value_w1, self.value_b1)
        v = [self.relu(x) for x in v]
        value_raw = self.value_b2
        for x, w in zip(v, self.value_w2):
            value_raw += x * w
        value = self.sigmoid(value_raw)

        result = {
            "policy": policy_logits,
            "value": value,
        }

        if return_hidden:
            result["hidden"] = h

        if return_cache:
            result["cache"] = {
                "inputs": inputs,
                "h": h,
                "policy": policy_logits,
                "value_raw": value_raw,
                "value": value,
            }

        return result

    def parameter_count(self):
        total = self.CHANNELS * self.INPUT_SIZE + self.CHANNELS
        for block in self.blocks:
            total += self.CHANNELS * self.CHANNELS * 2 + self.CHANNELS * 2
        total += self.CHANNELS * 512 + 512
        total += 512 * self.POLICY_SIZE + self.POLICY_SIZE
        total += self.CHANNELS * 64 + 64
        total += 64 + 1
        return total
