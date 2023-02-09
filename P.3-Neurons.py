import numpy as np

# A neuron with inputs, weights, and a bias
inputs = [1, 2, 3, 2.5]

# Weights for the input values
weights = [ [0.2, 0.8, -0.5, 1.0]
            [0.5, -0.91, 0.26, -0.5] 
            [-0.26, -0.27, 0.17, 0.87] ]

biases = [2, 3, 0.5]

# Vector - 1d
# Matrix - multidimensonal, rectangular
# Can dot a matrix * vector but not vector * matrix

# Dot product: [a_0 * w_0 + a_1 + w_1 ..., b_0 * w_0 + b_1 * w_1 ..., c_0 * w_0 + c_1 * w_1 ...]

output = np.dot(weights, inputs) + biases
print(output)

""" layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias, in zip(weights, biases):
    neuron_output = 0 # Output of given neuron
    """ for n_input, weight in zip(inputs, neuron_weights):
        neruon_output += n_input * weight """
    neruon_output = np.dot(neuron_weights, inputs) + neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs) """