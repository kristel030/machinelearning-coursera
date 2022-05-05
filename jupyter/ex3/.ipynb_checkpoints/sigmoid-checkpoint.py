import numpy as np

# the sigmoid function should work vor scalar values, vectors (1D arrays) and matrices (2D arrays)
def sigmoid(x):
  
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig