#neuron example
inputs = [1, 2, 3, 2.5]
weight = [0.2, 0.8, -0.5, 1.0]
bias = 2




#all inputs * weights + bias
output = inputs[0]*weight[0] + inputs[1]*weight[1] + inputs[2]*weight[2] + inputs[3]*weight[3] + bias
print(output)