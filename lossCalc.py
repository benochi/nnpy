# categorical cross-entropy, finding the precision of a NN.
# negative sum of target value, multiplied by the log of the predicted value,
# for each of the values in distribution.
# one hot coding. Vector of n classes long - filled with 0's except at index of target class it will be 1. ie [1,0,0]
# IE: classes 5, label 4 = [0,0,0,0,1] 4 is index, 5 is length.
# Many ways to calculate loss, often customized.
# Natural logarithm: y = loge^x = ln(x)  e is eulers number.
# natural log is base e.
# logarithm is solving for x in e ** x = b
# HOW IT WORKS:
# import numpy as np
# import math

# b = 5.2
# print(np.log(b))  # solves for Eulers number ** output = b.
# print(math.e**1.6486586255873816)  # 5.19 repeating do to programming issue.
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]
# target_class = 0 #index 0 would be hot

loss = -(
    math.log(softmax_output[0]) * target_output[0]
    + math.log(softmax_output[1]) * target_output[1]
    + math.log(softmax_output[2]) * target_output[2]
)

print(loss)
# same as loss = -math.log(softmax_output[0])
loss2 = -math.log(softmax_output[0])
print(loss2)

print(-math.log(0.7))
print(-math.log(0.5))
# confidence(how accurate its prediction is) gets higher the further is moves from the target of 1.
# so .5 is a bigger loss than .07
