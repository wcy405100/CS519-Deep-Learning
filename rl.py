import numpy as np
import scipy

# After we get the result from calcu()
# use ReLu in the whole matrix
# y is the gradient of ReLu
def ReLu(x):
    temp = (x+abs(x))/2
    y = scipy.sign(temp)
    return temp, np.array(y)