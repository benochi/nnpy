#neuron example
import numpy as np
inputs = [1, 2, 3, 2.5] #inputs don't change, you change weights.  inputs represent data passed into NN. 
weights = [
            [0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.87]
          ]

biases = [2, 3, 0.5]


output = np.dot(weights, inputs) + biases #if putting inputs first will get shape error. 

print(output)





# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
#   neuron_output = 0
#   for n_input, weight in zip(inputs, neuron_weights):
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# print(layer_outputs)