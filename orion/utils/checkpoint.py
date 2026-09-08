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

            "w1": network.w1,
            "b1": network.b1,

            "w2": network.w2,
            "b2": network.b2,

            "policy_w": network.policy_w,
            "policy_b": network.policy_b,

            "value_w": network.value_w,
            "value_b": network.value_b
        }

        with open(path, "w") as file:
            json.dump(data, file)

        return path

    def load(self, network, path):

        with open(path, "r") as file:
            data = json.load(file)

        network.w1 = data["w1"]
        network.b1 = data["b1"]

        network.w2 = data["w2"]
        network.b2 = data["b2"]

        network.policy_w = data["policy_w"]
        network.policy_b = data["policy_b"]

        network.value_w = data["value_w"]
        network.value_b = data["value_b"]

        return data["episode"]
