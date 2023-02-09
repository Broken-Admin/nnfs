import numpy as np
import nnfs
# Use same dataset in order to reproduce results
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

# Neural Networks from Scratch, produce reproducible results
nnfs.init()

# Inputs
# 100 points, 3 classes of spirals
# Inputs, targets
X, y = spiral_data(100, 3)

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

""" # Rectify linear activation function on an array
for i in inputs:
    # Output the maximum value between 0 and i
    output.append(max(0, i))
print(output) """

# Rectify linear activation function
class Activation_ReLU:
    def forward_pass(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward_pass(X)
activation1.forward_pass(layer1.output)

print(activation1.output)