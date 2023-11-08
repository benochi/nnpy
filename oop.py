import numpy as np

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]  # 3 samples of input data


# when loading a model, just saving weights and biases. with new NN need to initialize weights and biases - weights random values between 1 and -1, tighter range is better.
# tighter range means thing should tend between 1, -1 so input values will hopefully not explode the range. -0.1 - 0.1 for weights, biases tend to be initialized as zero.
class Layer_Dense:
    def __init__(self):
        pass

    def forward(self):
        pass
