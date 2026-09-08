import math


class PolicyTrainer:

    def __init__(self, network, learning_rate=0.01):
        self.network = network
        self.learning_rate = learning_rate

    def softmax(self, scores):

        maximum = max(scores)

        exp_scores = [
            math.exp(x - maximum)
            for x in scores
        ]

        total = sum(exp_scores)

        return [
            x / total
            for x in exp_scores
        ]

    def train_sample(self, hidden, policy_logits, target_action):

        probabilities = self.softmax(policy_logits)

        target_probability = probabilities[target_action]

        loss = -math.log(
            max(target_probability, 1e-12)
        )

        # Gradient of softmax + cross entropy.
        gradient = probabilities[:]
        gradient[target_action] -= 1.0

        # Update policy head.
        for action in range(len(gradient)):

            g = gradient[action]

            for i in range(len(hidden)):

                self.network.policy_w[action][i] -= (
                    self.learning_rate *
                    g *
                    hidden[i]
                )

            self.network.policy_b[action] -= (
                self.learning_rate * g
            )

        return loss
