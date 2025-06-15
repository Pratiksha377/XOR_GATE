import numpy as np

#np.random.seed(42)


class NeuralNetwork:
    def __init__(self, alpha):
        self.alpha = alpha
        self.W1 = np.random.randn(2, 2) * np.sqrt(2 / 2)
        print("W1:\n", self.W1)
        self.W2 = np.random.randn(2, 1) * np.sqrt(2 / 2)
        self.b1 = np.zeros((1, 2))
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

    def train(self, X, y, epochs=600000):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if i % 100000 == 0:

                loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
                print(f"Epoch {i} - Loss: {loss:.4f}")

    def accuracy(self, X, y):
        output = self.forward(X)
        predictions = np.round(output)
        correct = np.sum(predictions == y)
        return correct / y.shape[0]


# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(alpha=0.001)#also try with 0.01 and 0.1
nn.train(X, y)

print("\nOutput after training:\n", nn.forward(X))
print("Predictions:\n", np.round(nn.forward(X)))
print("Accuracy:", nn.accuracy(X, y))
