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

        # Calculate and save the cost J in every iteration
        cost = cc.computeCost(X, y, theta) 
        J_history[iter, 0] = cost;
        
        # Perform a single gradient step
        # --> vectorized implementation of 
        #     theta = theta - alpha * 1/m * sum(i=1:m): ( (h(x(i)) - y(i))^2 * x(i))
        #       where h(x)) = theta'*x

        # calculate the hypothesis values for the current theta
        h = X @ theta
        
        # slightly change (improve) theta for the next iteration, using the learning rate 'alpha' as one of the parameters
        theta = theta - (alpha * (h-y).T @ X * 1/m).T
        
    return theta, J_history