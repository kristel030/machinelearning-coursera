import numpy as np

# Task: 
# Implement the least squares cost function, 
#   based on the hypothesis h(x) = theta[0] + theta[1]*x[1]

def computeCost(X, y, theta):
    cost = 0
    
    # ====================== YOUR CODE HERE ======================
    
    # determine number of measurements
    m = y.shape[0]
    
    # calculate 'least squares' cost function
    # --> vectorized implementation of 1/2 * m * sum(i=1:m): ( (h(x(i)) - y(i))^2 )
    #       where h(x)) = theta'*x
    J = np.transpose(X @ theta - y) @ (X @ theta - y) / (2*m)
    
    # J is a single cell 2D matrix, get the value of that single cell
    cost = J[0,0]

    # ============================================================
    
    return cost