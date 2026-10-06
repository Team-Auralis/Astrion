import math

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ReinforcementTrainer:

    def __init__(
        self,
        network,
        learning_rate=0.0001,
        gradient_clip=1.0
    ):
        self.network = network
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self._torch_model = None

        if HAS_TORCH:
            try:
                self._build_torch_model()
            except Exception:
                pass

    def _build_torch_model(self):
        from train_fast import OrionResNet
        model = OrionResNet()
        self._load_weights_into_torch(model)
        self._torch_model = model
        self._torch_optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate
        )

    def _load_weights_into_torch(self, model):
        with torch.no_grad():
            model.input_fc.weight.data = torch.tensor(
                self.network.input_w, dtype=torch.float32
            )
            model.input_fc.bias.data = torch.tensor(
                self.network.input_b, dtype=torch.float32
            )
            for i, block in enumerate(self.network.blocks):
                if i >= len(model.blocks):
                    break
                model.blocks[i].fc1.weight.data = torch.tensor(
                    block.w_a, dtype=torch.float32
                )
                model.blocks[i].fc1.bias.data = torch.tensor(
                    block.b_a, dtype=torch.float32
                )
                model.blocks[i].fc2.weight.data = torch.tensor(
                    block.w_b, dtype=torch.float32
                )
                model.blocks[i].fc2.bias.data = torch.tensor(
                    block.b_b, dtype=torch.float32
                )
            model.policy_fc1.weight.data = torch.tensor(
                self.network.policy_w1, dtype=torch.float32
            )
            model.policy_fc1.bias.data = torch.tensor(
                self.network.policy_b1, dtype=torch.float32
            )
            model.policy_fc2.weight.data = torch.tensor(
                self.network.policy_w2, dtype=torch.float32
            )
            model.policy_fc2.bias.data = torch.tensor(
                self.network.policy_b2, dtype=torch.float32
            )
            model.value_fc1.weight.data = torch.tensor(
                self.network.value_w1, dtype=torch.float32
            )
            model.value_fc1.bias.data = torch.tensor(
                self.network.value_b1, dtype=torch.float32
            )
            model.value_fc2.weight.data = torch.tensor(
                [self.network.value_w2], dtype=torch.float32
            )
            model.value_fc2.bias.data = torch.tensor(
                [self.network.value_b2], dtype=torch.float32
            )

    def _sync_from_torch(self, model):
        with torch.no_grad():
            self.network.input_w = model.input_fc.weight.cpu().numpy().tolist()
            self.network.input_b = model.input_fc.bias.cpu().numpy().tolist()
            for i, block in enumerate(self.network.blocks):
                if i >= len(model.blocks):
                    break
                block.w_a = model.blocks[i].fc1.weight.cpu().numpy().tolist()
                block.b_a = model.blocks[i].fc1.bias.cpu().numpy().tolist()
                block.w_b = model.blocks[i].fc2.weight.cpu().numpy().tolist()
                block.b_b = model.blocks[i].fc2.bias.cpu().numpy().tolist()
            self.network.policy_w1 = model.policy_fc1.weight.cpu().numpy().tolist()
            self.network.policy_b1 = model.policy_fc1.bias.cpu().numpy().tolist()
            self.network.policy_w2 = model.policy_fc2.weight.cpu().numpy().tolist()
            self.network.policy_b2 = model.policy_fc2.bias.cpu().numpy().tolist()
            self.network.value_w1 = model.value_fc1.weight.cpu().numpy().tolist()
            self.network.value_b1 = model.value_fc1.bias.cpu().numpy().tolist()
            self.network.value_w2 = model.value_fc2.weight.squeeze(0).cpu().numpy().tolist()
            self.network.value_b2 = float(model.value_fc2.bias.squeeze().item())

    def softmax(self, logits):
        maximum = max(logits)
        values = [math.exp(x - maximum) for x in logits]
        total = sum(values)
        return [x / total for x in values]

    def masked_softmax(self, logits, legal_actions):
        if not legal_actions:
            raise ValueError("No legal actions supplied")

        probabilities = [0.0] * len(logits)
        maximum = max(logits[action] for action in legal_actions)

        values = {}
        total = 0.0
        for action in legal_actions:
            value = math.exp(logits[action] - maximum)
            values[action] = value
            total += value

        for action in legal_actions:
            probabilities[action] = values[action] / total

        return probabilities

    def clip(self, value):
        if value > self.gradient_clip:
            return self.gradient_clip
        if value < -self.gradient_clip:
            return -self.gradient_clip
        return value

    def train_supervised_sample(self, cache, target_action, legal_actions):
        current_value = cache["value"]
        return self.train_sample(
            cache=cache,
            action=target_action,
            reward=1.0,
            value_target=current_value,
            legal_actions=legal_actions
        )

    def train_sample(self, cache, action, reward, value_target, legal_actions):
        if self._torch_model is not None:
            return self._train_torch(cache, action, reward, value_target, legal_actions)
        return self._train_pure_python(cache, action, reward, value_target, legal_actions)

    def _train_torch(self, cache, action, reward, value_target, legal_actions):
        inputs = cache["inputs"]
        x = torch.tensor([inputs], dtype=torch.float32)

        self._torch_model.eval()
        self._torch_optimizer.zero_grad()

        policy_logits, value_pred = self._torch_model(x)

        target = torch.tensor([action], dtype=torch.long)
        policy_loss = nn.functional.cross_entropy(policy_logits, target)

        value_target_t = torch.tensor([[value_target]], dtype=torch.float32)
        value_loss = nn.functional.mse_loss(value_pred, value_target_t)

        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._torch_model.parameters(), 1.0)
        self._torch_optimizer.step()

        self._sync_from_torch(self._torch_model)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item(),
        }

    def _train_pure_python(self, cache, action, reward, value_target, legal_actions):
        inputs = cache["inputs"]
        h = cache.get("h", cache.get("h2", []))
        policy_logits = cache["policy"]
        prediction = cache["value"]

        probabilities = self.masked_softmax(policy_logits, legal_actions)
        selected_probability = max(probabilities[action], 1e-12)
        policy_loss = -reward * math.log(selected_probability)

        d_policy = [0.0] * len(policy_logits)
        for move in legal_actions:
            target_val = 1.0 if move == action else 0.0
            d_policy[move] = reward * (probabilities[move] - target_val)

        value_error = prediction - value_target
        value_loss = value_error * value_error
        d_value_raw = 2.0 * value_error * prediction * (1.0 - prediction)
        d_value_raw = self.clip(d_value_raw)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": policy_loss + value_loss,
        }
