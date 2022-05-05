import numpy as np

# Task: 
# Implement the least squares cost function, 
#   based on the hypothesis h(x) = theta[0] + theta[1]*x[1]

def computeCost(X, y, theta):
    cost = 0
    
    # reshape y and theta from a 1D array to a single column 2D array
    # this makes computations easier
    y = y.reshape(y.shape[0],-1)
    theta = theta.reshape(theta.shape[0], -1)
    
    # ====================== YOUR CODE HERE ======================
    
    # determine number of measurements
    m = y.shape[0]
    
    # calculate 'least squares' cost function
    # --> vectorized implementation of 1/2 * m * sum(i=1:m): ( (h(x(i)) - y(i))^2 )
    #       where h(x)) = theta'*x
    #J = np.transpose(X @ theta - y) @ (X @ theta - y) / (2*m)
    
    h = X @ theta
    J = (h - y).T @ (h - y) / (2*m)
    
    # J is a single cell 2D matrix, get the value of that single cell
    cost = J[0,0]

    # ============================================================
    
    return cost