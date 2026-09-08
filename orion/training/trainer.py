class ORIONTrainer:

    def __init__(self, network, learning_rate=0.01):

        self.network = network
        self.learning_rate = learning_rate

    def value_loss(self, prediction, target):

        error = prediction - target

        return error * error

    def train_value_sample(self, hidden, prediction, target):

        error = prediction - target

        # Sigmoid derivative
        gradient = error * prediction * (1.0 - prediction)

        # Update value head
        for i in range(len(self.network.value_w)):

            self.network.value_w[i] -= (
                self.learning_rate *
                gradient *
                hidden[i]
            )

        self.network.value_b -= (
            self.learning_rate *
            gradient
        )

        return self.value_loss(
            prediction,
            target
        )
