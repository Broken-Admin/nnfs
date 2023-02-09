import math

softmax_output = [0.7, 0.1, 0.2]


# Target is class 0
# Equation for "loss" or error in result
# that increases exponentially due to the capping
# at 1.00, min is 0.00
loss = -math.log(softmax_output[0])
print(loss)
