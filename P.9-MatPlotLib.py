import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

input_values, expected_class = vertical_data(samples=100, classes=3)

# Neural Network is to take each point, a list of 2 values,
# in the 2d array input_values and predict it's class
# given the expected classifications

# input_values[:, 0], takes all arrays inside of the 2d array
# and tells to take the 0th value
# input_values[:, 0], takes all arrays inside of the 2d array
# and tells to take the 1st value

# cmap is 'blg' as in 3 bit assignment of blue, red, green,
# as the expected classes are 0, 1, 2

# s is size of dot

plt.scatter(input_values[:, 0], input_values[:, 1], c=expected_class, s=15, cmap='brg')
plt.show()