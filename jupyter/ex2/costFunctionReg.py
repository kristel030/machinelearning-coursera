from numpy import array, zeros, log
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, llambda):
    """
    Compute the cost of a particular choice of theta.
    :param theta: 1D array, size n
    :param X: 2D array, size n x m
    :param y: 1D array, size m
    :param lambda: ???
    :return: cost (scalar value) and grad (1D array, size n)    """
    
    # Verify dimensions of input parameters
    if (theta.ndim != 1): 
        raise AssertionError ('theta is a ' + str(theta.ndim) + 'D array; expected a 1D array')
    if (X.ndim != 2): 
        raise AssertionError ('X is a ' + str(X.ndim) + 'D array; expected a 2D array') 
    if (y.ndim != 1): 
        raise AssertionError ('y is a ' + str(y.ndim) + 'D array; expected a 1D array') 
        
    # Initialize some useful values
    m = y.shape[0] # number of training examples    
        
    # You need to calculate the following variables correctly 
    J = 0
    grad = zeros(m)
    
    # reshape y and theta from a 1D array to a single column 2D array
    # this makes computations easier
    y = y.reshape(y.shape[0],-1)
    theta = theta.reshape(theta.shape[0], -1)
      
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #                 derivatives of the cost w.r.t. each parameter in theta
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X @ theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations. 
    # Note: J should be a scalar value, grad should have the same dimensions as theta.
    
    # helper variable to compute J
    h = sigmoid(X @ theta);
    
    # compute J in two parts
    theta_without0 = theta[1:]
    base_J = (-(y.T) @ log(h) - (1-y).T @ log(1-h)) / m 
    regularized_J = llambda / (2*m) * theta_without0.T @ theta_without0
    _J = base_J + regularized_J;
    J = _J[0, 0] # _J is a 1 x 1 matrix, take the value of the single cell
    
    # compute grad
    base_grad = X.T @ (sigmoid(X @ theta) - y) / m  
    regularized_grad = (llambda/m) * theta
    _grad = base_grad + regularized_grad
    
    grad0 = sum (h-y) / m;
    _grad[0,0] = grad0;
    
    grad = _grad[:, 0] # _grad is a 3 x 1 matrix, we need its single column as a 1D vector 
    
    # ============================================================
    
    # Verify dimensions of output values
    if (J.ndim != 0): 
        raise AssertionError ('J is a ' + str(J.ndim) + 'D array; expected a scalar value')
    if (grad.ndim != 1): 
        raise AssertionError ('grad is a ' + str(X.ndim) + 'D array; expected a 1D array')     
    
    return J, grad