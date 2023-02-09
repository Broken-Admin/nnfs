import numpy as np
import nnfs
# Use same dataset in order to reproduce results
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

# Neural Networks from Scratch, produce reproducible results
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Random Gaussian distribution around zero, scaled down 0 to 1
        # Avoid transposing by reversing the size to be INPUTS x NEURONS
        # Rather than the other way done as previously
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

""" # Rectify linear activation function on an array
for i in inputs:
    # Output the maximum value between 0 and i
    output.append(max(0, i))
print(output) """

# Rectify linear activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Softmax activation function
class Activation_Softmax:
    def forward(self, inputs):
        # Normalize the values to reduce large numbers
        # and make the largest number 0
        # With respect to the batch format
        inputs -= np.max(inputs, axis=1, keepdims=True)
        # The maximum exponential value will now be
        # e to the 0 or 1
        exp_values = np.exp(inputs)
        # Sum the rows, horizontally, while keeping dimensions
        probabilities_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities_values

X, y = spiral_data(samples=100, classes=3)

# 2 inputs, 3 nodes
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Output layer, 3 output class types
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Each array is equal to a batch, 3 neuron values
print(activation2.output)