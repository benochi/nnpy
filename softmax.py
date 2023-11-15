import math

# softmax is used for output layer
layer_outputs = [4.8, 1.2, 2.385]
# exponential function helps with ReLU  y = e to power of x.
# e = 2.718281828459045 (Eulers number) help prevent -1 and 1 giving same output or
# negativity causing bad output data.

Eulers = 2.718281828459045
E = math.e  # same as hardcoding.

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)
