import math


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

    def softmax(self, logits):

        maximum = max(logits)

        values = [
            math.exp(x - maximum)
            for x in logits
        ]

        total = sum(values)

        return [
            x / total
            for x in values
        ]

    def masked_softmax(self, logits, legal_actions):

        if not legal_actions:
            raise ValueError(
                "No legal actions supplied"
            )

        probabilities = [0.0] * len(logits)

        maximum = max(
            logits[action]
            for action in legal_actions
        )

        values = {}
        total = 0.0

        for action in legal_actions:

            value = math.exp(
                logits[action] - maximum
            )

            values[action] = value
            total += value

        for action in legal_actions:

            probabilities[action] = (
                values[action] / total
            )

        return probabilities

    def clip(self, value):

        if value > self.gradient_clip:
            return self.gradient_clip

        if value < -self.gradient_clip:
            return -self.gradient_clip

        return value

    def train_supervised_sample(
        self,
        cache,
        target_action,
        legal_actions
    ):
        """
        Supervised policy training.

        The target action is the correct move supplied
        by the chess puzzle dataset.

        The existing policy-gradient implementation is
        reused with reward=1.0, which produces standard
        masked softmax cross-entropy.

        The value target equals the current prediction,
        producing zero value gradient. Therefore this
        method trains the policy only.
        """

        current_value = cache["value"]

        return self.train_sample(
            cache=cache,
            action=target_action,
            reward=1.0,
            value_target=current_value,
            legal_actions=legal_actions
        )

    def train_sample(
        self,
        cache,
        action,
        reward,
        value_target,
        legal_actions
    ):

        inputs = cache["inputs"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        z2 = cache["z2"]
        h2 = cache["h2"]

        logits = cache["policy"]
        prediction = cache["value"]

        # ==============================================
        # POLICY GRADIENT
        # ==============================================

        probabilities = self.masked_softmax(
            logits,
            legal_actions
        )

        selected_probability = max(
            probabilities[action],
            1e-12
        )

        policy_loss = -reward * math.log(
            selected_probability
        )

        d_policy = [0.0] * len(logits)

        for move in legal_actions:

            target = (
                1.0
                if move == action
                else 0.0
            )

            d_policy[move] = (
                reward *
                (
                    probabilities[move]
                    - target
                )
            )

        # ==============================================
        # VALUE GRADIENT
        # ==============================================

        value_error = (
            prediction -
            value_target
        )

        value_loss = (
            value_error *
            value_error
        )

        d_value_raw = (
            2.0 *
            value_error *
            prediction *
            (1.0 - prediction)
        )

        d_value_raw = self.clip(
            d_value_raw
        )

        # ==============================================
        # GRADIENT BUFFERS
        # ==============================================

        d_h2 = [0.0] * len(h2)
        d_h1 = [0.0] * len(h1)

        d_z2 = [0.0] * len(z2)
        d_z1 = [0.0] * len(z1)

        # ==============================================
        # POLICY HEAD GRADIENTS
        # ==============================================

        policy_gradients = {}

        for move in legal_actions:

            gradient = self.clip(
                d_policy[move]
            )

            policy_gradients[move] = gradient

            row = self.network.policy_w[
                move
            ]

            for i in range(len(h2)):

                d_h2[i] += (
                    gradient *
                    row[i]
                )

        # ==============================================
        # VALUE HEAD GRADIENT
        # ==============================================

        original_value_w = (
            self.network.value_w[:]
        )

        for i in range(len(h2)):

            d_h2[i] += (
                d_value_raw *
                original_value_w[i]
            )

        # ==============================================
        # RELU 2
        # ==============================================

        for i in range(len(z2)):

            if z2[i] > 0.0:

                d_z2[i] = d_h2[i]

        # ==============================================
        # W2 GRADIENT
        # ==============================================

        w2_gradients = [
            [0.0] * len(h1)
            for _ in range(len(z2))
        ]

        b2_gradients = [
            0.0
            for _ in range(len(z2))
        ]

        original_w2 = [
            row[:]
            for row in self.network.w2
        ]

        for i in range(len(z2)):

            gradient = self.clip(
                d_z2[i]
            )

            b2_gradients[i] = gradient

            for j in range(len(h1)):

                w2_gradients[i][j] = (
                    gradient *
                    h1[j]
                )

                d_h1[j] += (
                    gradient *
                    original_w2[i][j]
                )

        # ==============================================
        # RELU 1
        # ==============================================

        for i in range(len(z1)):

            if z1[i] > 0.0:

                d_z1[i] = d_h1[i]

        # ==============================================
        # W1 GRADIENT
        # ==============================================

        w1_gradients = [
            [0.0] * len(inputs)
            for _ in range(len(z1))
        ]

        b1_gradients = [
            0.0
            for _ in range(len(z1))
        ]

        for i in range(len(z1)):

            gradient = self.clip(
                d_z1[i]
            )

            b1_gradients[i] = gradient

            for j in range(len(inputs)):

                w1_gradients[i][j] = (
                    gradient *
                    inputs[j]
                )

        # ==============================================
        # APPLY ALL UPDATES
        # ==============================================

        # Policy
        for move in legal_actions:

            gradient = policy_gradients[
                move
            ]

            row = self.network.policy_w[
                move
            ]

            for i in range(len(h2)):

                row[i] -= (
                    self.learning_rate *
                    gradient *
                    h2[i]
                )

            self.network.policy_b[
                move
            ] -= (
                self.learning_rate *
                gradient
            )

        # Value
        for i in range(len(h2)):

            self.network.value_w[i] -= (
                self.learning_rate *
                d_value_raw *
                h2[i]
            )

        self.network.value_b -= (
            self.learning_rate *
            d_value_raw
        )

        # W2
        for i in range(len(z2)):

            for j in range(len(h1)):

                self.network.w2[i][j] -= (
                    self.learning_rate *
                    w2_gradients[i][j]
                )

            self.network.b2[i] -= (
                self.learning_rate *
                b2_gradients[i]
            )

        # W1
        for i in range(len(z1)):

            for j in range(len(inputs)):

                self.network.w1[i][j] -= (
                    self.learning_rate *
                    w1_gradients[i][j]
                )

            self.network.b1[i] -= (
                self.learning_rate *
                b1_gradients[i]
            )

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": (
                policy_loss +
                value_loss
            )
        }
