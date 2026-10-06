import json
from pathlib import Path


class Checkpoint:

    def __init__(self, directory="checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(self, network, episode):

        path = self.directory / f"orion_{episode}.json"

        data = {
            "episode": episode,
            "arch": "resnet",
            "channels": network.CHANNELS,
            "num_blocks": network.NUM_BLOCKS,
            "input_size": network.INPUT_SIZE,

            "input_w": network.input_w,
            "input_b": network.input_b,

            "block_wa": [b.w_a for b in network.blocks],
            "block_ba": [b.b_a for b in network.blocks],
            "block_wb": [b.w_b for b in network.blocks],
            "block_bb": [b.b_b for b in network.blocks],

            "policy_w1": network.policy_w1,
            "policy_b1": network.policy_b1,
            "policy_w2": network.policy_w2,
            "policy_b2": network.policy_b2,

            "value_w1": network.value_w1,
            "value_b1": network.value_b1,
            "value_w2": network.value_w2,
            "value_b2": network.value_b2,
        }

        with open(path, "w") as file:
            json.dump(data, file)

        return path

    def load(self, network, path):

        with open(path, "r") as file:
            data = json.load(file)

        if data.get("arch") == "resnet":
            return self._load_resnet(network, data)
        else:
            return self._load_legacy(network, data)

    def _load_resnet(self, network, data):
        from orion.model.network import ResidualBlock

        network.CHANNELS = data.get("channels", 256)
        network.NUM_BLOCKS = data.get("num_blocks", 6)
        network.INPUT_SIZE = data.get("input_size", 1152)

        network.input_w = data["input_w"]
        network.input_b = data["input_b"]

        network.blocks = []
        for i in range(len(data["block_wa"])):
            block = ResidualBlock.__new__(ResidualBlock)
            block.w_a = data["block_wa"][i]
            block.b_a = data["block_ba"][i]
            block.w_b = data["block_wb"][i]
            block.b_b = data["block_bb"][i]
            network.blocks.append(block)

        network.policy_w1 = data["policy_w1"]
        network.policy_b1 = data["policy_b1"]
        network.policy_w2 = data["policy_w2"]
        network.policy_b2 = data["policy_b2"]

        network.value_w1 = data["value_w1"]
        network.value_b1 = data["value_b1"]
        network.value_w2 = data["value_w2"]
        network.value_b2 = data["value_b2"]

        return data["episode"]

    def _load_legacy(self, network, data):
        """Load old 2-layer MLP checkpoint for backward compatibility."""
        from orion.model.network import ResidualBlock

        network.CHANNELS = 128
        network.NUM_BLOCKS = 0
        network.INPUT_SIZE = 896
        network.blocks = []

        network.input_w = data["w1"]
        network.input_b = data["b1"]

        network.policy_w1 = data["w2"]
        network.policy_b1 = data["b2"]
        network.policy_w2 = data["policy_w"]
        network.policy_b2 = data["policy_b"]

        network.value_w1 = []
        network.value_b1 = []
        network.value_w2 = data["value_w"]
        network.value_b2 = data["value_b"]

        return data["episode"]
