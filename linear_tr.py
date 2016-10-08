import numpy as np
# calculate the W*x+b value before the activation function
# the return dimension will be (hidden unit size) * (image size)
def calcu(x,w,b):
    temp = np.dot(x,w)
    for i in temp:
        i += b
    return np.array(temp)

