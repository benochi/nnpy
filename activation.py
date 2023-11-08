# ACTIVATION Functions apply after inputs * weights + bias we use them because without input = output, and is linear. So they won't handle non-linear data(such as a sine wave)
# step function: if input is greater than 0 output = 1 otherwise 0
# sigmoid activation function(more reliable than step function) helps calculate how wrong the NN is. Then optimize.
# rectified linear activation function, if output is greater than 0 then x = x, if less than 0, x = 0.
# sigmoid has an issue called vanishing gradient that RL doesn't have.
# RL is fast, sigmoid is slower
# RL can be tweaked by moving bias. if we negate weight it flips RL to determine where it deactivates.
# if adjusting bias of second neuron RL will become offset vertically. and weight for neuron 2 will provide upper and lower bound. like a _/- type shape.
import numpy as np

np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
# rectified linear activation
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
# also RLA without elif statement:
# for i in inputs:
#     output.append(max(0, i))

print(output)

# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))

#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases

# layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)

# layer1.forward(X)
# print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)
