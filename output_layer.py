import numpy as np

# use output function to classify the layer2 to {0,1}
def output(layer2):
    outpu = [0 if temp < 0.5 else 1 for temp in layer2]
    return np.array(outpu)