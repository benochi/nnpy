import math
import numpy as np
import nnfs

# batch = easier, uncomment for non batch
# iterating over 2d matrix and handling 3 sums.
nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)  # get exponential values

# print(exp_values)
# norm_values = exp_values / np.sum(exp_values)

# axis 0 is sum of columns, 1 for rows(waht we want)  keepdims makes same dimensions.
# print(np.sum(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

# exponential values will explode size of numbers and overflow.
# solved by subtracting max value from other values.


# print(norm_values)
# print(sum(norm_values))

# E = math.e  # same as hardcoding.
# INPUT -> EXPONENTIATE -> NORMALIZE -> OUTPUT
# commented out not using numpy vs numpy.
# softmax is EXPONENTIATE and NORMALIZE(formula Si,j = e^zi,j / Epsilon^Ll=1 e^zi,j)
# exponential function helps with ReLU  y = e to power of x.
# e = 2.718281828459045 (Eulers number) help prevent -1 and 1 giving same output or
# negativity causing bad output data.
# want to ditch negatives but not lose meaning.

# Eulers = 2.718281828459045
# exp_values = []

# for output in layer_outputs:
#     exp_values.append(E**output)

# print(exp_values)
# # normalize exponentiate values.
# norm_base = sum(exp_values)
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)

# print(norm_values)
# print(sum(norm_values))
