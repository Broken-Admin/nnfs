import numpy as np

# Batch outputs, multiple outputs from multiple single inputs
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])

# Indexes to target, actual data classification
class_targets = [0, 1, 1]

# Confidence in the correct target
# or likelyhood that it is the correct choice
# Fetch each value at each target index
confidence_values = softmax_outputs[
    range(len(softmax_outputs)), class_targets
]

# Print the confidence of each index
print(confidence_values)

# Can be used to fetch the values for loss calculation
# Calculate loss or inaccuracy for each
loss_values = -np.log(confidence_values)
print(loss_values)

# Calculate average loss
average_loss = np.mean(loss_values)
print(average_loss)

# Accuracy calculations
predictions = np.argmax(softmax_outputs, axis=1)
print(predictions)
accuracy = np.mean(predictions == class_targets)
print(accuracy)