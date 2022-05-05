import numpy as np
from sigmoid import sigmoid

def sigmoidGradient(z):
    """
    SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z
    
    g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function evaluated at z. 
    This should work regardless if z is a matrix or a vector. 
    In particular, if z is a vector or matrix, you should return the gradient for each element.
    """
    
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).

    g = np.multiply( sigmoid(z) , np.subtract(np.ones(z.shape) , sigmoid(z) ) )

    # =============================================================
    
    return g