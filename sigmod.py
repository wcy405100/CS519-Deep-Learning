import numpy as np

# sigmod function
def sigmod(x):
    return np.nan_to_num(1.0/(1.0+np.exp(-x)))