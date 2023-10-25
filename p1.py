#neuron example
inputs = [1, 2, 3, 2.5] #inputs don't change, you change weights.  inputs represent data passed into NN. 

weight1 = [0.2, 0.8, -0.5, 1.0]
weight2 = [0.5, -0.91, 0.26, -0.5]
weight3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5


#all inputs * weights + bias
output = [inputs[0]*weight1[0] + inputs[1]*weight1[1] + inputs[2]*weight1[2] + inputs[3]*weight1[3] + bias1,
          inputs[0]*weight2[0] + inputs[1]*weight2[1] + inputs[2]*weight2[2] + inputs[3]*weight2[3] + bias2,
          inputs[0]*weight3[0] + inputs[1]*weight3[1] + inputs[2]*weight3[2] + inputs[3]*weight3[3] + bias3]
print(output)