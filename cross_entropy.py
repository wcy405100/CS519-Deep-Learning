import numpy as np


# this funciton calculate the cross entropy of the prediction from the softmax layer
def cross_en(layer2, ground_t):
    layer2 = np.array(layer2)
    ground_t = np.squeeze(ground_t)
    entropy = -(ground_t * np.log(layer2)) - (1 - ground_t) * np.log(1 - layer2)
    return np.nan_to_num(np.mean(entropy))
