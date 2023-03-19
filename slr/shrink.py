import numpy as np

def shrink(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x)-alpha,0)