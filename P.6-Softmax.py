import numpy as np
import nnfs

nnfs.init()

## LINEAR SOFTMAX

layer_outputs = [4.8, 1.21, 2.385]

# Softmax activation, generally used to provide
# confidence probabilities for the output neurons

# Superceded by np.exp(...)
""" 
# Take Euler's constant to the inputs
# in order to bring all values positive
def exp_values(inputs):
    outputs = []
    for value in inputs:
        outputs.append(math.e**value)
    return(outputs)
"""

# Normalize the values between zero and one using the sum
# of all values
def normalize_values(inputs):
    norm_base = np.sum(inputs)
    norm_values = []
    for value in inputs:
        norm_values.append(value / norm_base)
    return(norm_values)


print(normalize_values(np.exp(layer_outputs)))

# INPUT -> SOFTMAX (EXPONENTIATE -> NORMALIZE) -> OUTPUT

## BATCH SOFTMAX
batch_outputs = [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]

def batch_softmax(inputs):
    exp_values = np.exp(inputs)
    # Sum the rows, horizontally, while keeping dimensions
    normalized_values = exp_values / np.sum(inputs, axis=1, keepdims=True)
    return(normalized_values)

print(batch_softmax(batch_outputs))