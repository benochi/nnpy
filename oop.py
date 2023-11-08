import numpy as np

np.random.seed(0)
# 3 samples of input data
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]


# when loading a model, just saving weights and biases. with new NN need to initialize weights and biases - weights random values between 1 and -1, tighter range is better.
# tighter range means thing should tend between 1, -1 so input values will hopefully not explode the range. -0.1 - 0.1 for weights, biases tend to be initialized as zero.
# if getting all zeros, sometimes bias might need tweaked, or they will propagate all zeros and be "dead"
# need to know size of input coming in, and how many neurons we want to have. Size of single sample.
# we shape the weights to skip the transpose later.
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # first param is shape and tuple.

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# create using inputs and neurons(can make this however big you want)
# layer 2 must match output of layer 1.
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
