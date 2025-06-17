import numpy as np

#np.random.seed(42)


class NeuralNetwork:
    def __init__(self, alpha):
        self.alpha = alpha
        self.W1 = np.random.randn(2, 4) * np.sqrt(2.0 / 2)  
        #print("W1:\n", self.W1)
        self.W2 = np.random.randn(4, 1) * np.sqrt(2.0 / 4)
        self.b1 = np.zeros((1, 4))
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, x, deriv=False):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if deriv else sig

    def relu(self, x, deriv=False):
        if deriv:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        m = y.shape[0]
        delta2 = (output - y) * self.sigmoid(self.z2, deriv=True)
        delta1 = delta2.dot(self.W2.T) * self.relu(self.z1, deriv=True)

        self.W2 -= self.alpha * self.a1.T.dot(delta2) / m
        self.b2 -= self.alpha * np.sum(delta2, axis=0, keepdims=True) / m
        self.W1 -= self.alpha * X.T.dot(delta1) / m
        self.b1 -= self.alpha * np.sum(delta1, axis=0, keepdims=True) / m

    def calculate_loss(self, y, output):
       def binary_cross_entropy(y, output, epsilon=1e-12):
            output = np.clip(output, epsilon, 1. - epsilon)
            return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))


    def train(self, X, y, max_epochs=100000):
        best_loss = float('inf')
        patience_counter = 0
        patience = 10000  

        for epoch in range(max_epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 2000 == 0:  
                current_loss = self.calculate_loss(y, output)
                accuracy = self.accuracy(X, y)
                print(f"Epoch {epoch} - Loss: {current_loss:.4f} - Accuracy: {accuracy:.2f}")

                if current_loss < best_loss - 0.001:  # Need significant improvement
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 2000

                if accuracy == 1.0:
                    print(f"Perfect accuracy achieved at epoch {epoch}")
                    break

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def accuracy(self, X, y):
        output = self.forward(X)
        predictions = np.round(output)
        correct = np.sum(predictions == y)
        return correct / y.shape[0]


# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Test with optimized learning rates
learning_rates = [0.1, 0.3, 1.0]

for lr in learning_rates:
    print(f"\n{'='*50}")
    print(f"Testing with learning rate: {lr}")
    print(f"{'='*50}")

    nn = NeuralNetwork(alpha=lr)
    nn.train(X, y)

    final_output = nn.forward(X)
    predictions = np.round(final_output)
    accuracy = nn.accuracy(X, y)

    print(f"\nFinal Results for lr={lr}:")
    print("Raw Output:\n", final_output)
    print("Predictions:\n", predictions.astype(int))
    print("Expected:\n", y)
    print(f"Final Accuracy: {accuracy:.2f}")
