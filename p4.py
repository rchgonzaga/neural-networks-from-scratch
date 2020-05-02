import numpy as np

#layer1
inputs  = [
  [1,2,3,2.5],
  [2,5.7,2.6,1.2],
  [0.2,9.3,4.5,5.2]
]
weights = [
  [1.3, 1.6, -2.9, 1.2],
  [0.5, 1.13, -0.26, -5.5],
  [-0.26, -13.27, 0.7, 2.87]
]
biases    = [0.12, -1.2, 1.5]

# layer2
inputs2  = [
  [1,2,3,2.5],
  [2,5.7,2.6,1.2],
  [0.2,9.3,4.5,5.2]
]
weights2 = [
  [0.2, 0.8, -0.5, 1.0],
  [0.5, -0.91, 0.26, -0.5],
  [-0.26, -0.27, 0.17, 0.87]
]
biases2    = [2, 3, 0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)
