from numpy import exp

# the sigmoid function should work vor scalar values, vectors (1D arrays) and matrices (2D arrays)
def sigmoid(x):
    
    # your code goes here ---------------------------------
    # sig = 0
    z = exp(-x)
    sig = 1 / (1 + z)
    
    # end of your code ------------------------------------

    return sig