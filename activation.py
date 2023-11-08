# ACTIVATION Functions apply after inputs * weights + bias we use them because without input = output, and is linear. So they won't handle non-linear data(such as a sine wave)
# step function: if input is greater than 0 output = 1 otherwise 0
# sigmoid activation function(more reliable than step function) helps calculate how wrong the NN is. Then optimize.
# rectified linear activation function, if output is greater than 0 then x = x, if less than 0, x = 0.
# sigmoid has an issue called vanishing gradient that RL doesn't have.
# RL is fast, sigmoid is slower
# RL can be tweaked by moving bias. if we negate weight it flips RL to determine where it deactivates.
# if adjusting bias of second neuron RL will become offset vertically. and weight for neuron 2 will provide upper and lower bound. like a _/- type shape.
