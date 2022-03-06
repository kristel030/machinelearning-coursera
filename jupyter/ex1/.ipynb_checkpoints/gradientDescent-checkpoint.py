import numpy as np
import computeCost as cc

# Task: 
# Implement the gradient descent algorithm.

def gradientDescent(X, y, theta, alpha, iterations):
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    J_history = np.zeros((iterations, 1));

    for iter in range(0, iterations):

        #  ====================== YOUR CODE HERE ======================
        #  Instructions: Perform a single gradient step on the parameter vector
        #                theta. 
        # 
        #  Hint: While debugging, it can be useful to print out the values
        #        of the cost function (computeCost) and gradient here.
        # 

        cost = cc.computeCost(X, y, theta)
        # print('cost: ' + str(cost))
        
        # Save the cost J in every iteration  
        J_history[iter, 0] = cost;
        
        # perform a single gradient step
        # --> vectorized implementation of 
        #     theta = theta - alpha * 1/m * sum(i=1:m): ( (h(x(i)) - y(i))^2 * x(i))
        #       where h(x)) = theta'*x

        h = X @ theta
        # theta = theta - np.transpose((np.transpose(h-y) @ X * alpha * 1/m))
        theta = theta - ((h-y).T @ X * alpha * 1/m).T
        
    return theta, J_history
    
    
    ## Octave implementation:
	# h = X*theta;
	# theta = theta - ((h-y)' * X * alpha * 1/m)';