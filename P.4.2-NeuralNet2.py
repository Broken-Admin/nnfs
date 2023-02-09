import numpy as np

np.random.seed(0)

# Inputs
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Random Gaussian distribution around zero, scaled down 0 to 1
        # Avoid transposing by reversing the size to be INPUTS x NEURONS
        # Rather than the other way done as previously
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward_pass(X)
layer2.forward_pass(layer1.output)

print(layer2.output)