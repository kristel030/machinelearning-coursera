from numpy import append, zeros, ones, where
import scipy.optimize as opt
from lrCostFunction import lrCostFunction
from lrGradientDescent import lrGradientDescent
from lrCostGradientFunction import lrCostGradientFunction

def oneVsAll(X_data, y, num_labels, llambda):
    """
    [all_theta] = ONEVSALL(X, y, num_labels, llambda) trains num_labels logistic regression classifiers 
      and returns each of these classifiers in a matrix all_theta, 
      where the i-th row of all_theta corresponds to the classifier for label i
    """
    
    # Some useful variables
    m = X_data.shape[0]
    n = X_data.shape[1]

    # You need to return the following variables correctly 
    all_theta = zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = append( ones((m,1)), X_data, axis=1)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter llambda. 
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.
    #
    #       fmincg works similarly to fminunc, but is more efficient when we
    #       are dealing with large number of parameters.
    #
    # Example Code for fmincg:
    #
    #     % Set Initial theta
    #     initial_theta = zeros(n + 1, 1);
    #     
    #     % Set options for fminunc
    #     options = optimset('GradObj', 'on', 'MaxIter', 50);
    # 
    #     % Run fmincg to obtain the optimal theta
    #     % This function will return theta and the cost 
    #     [theta] = ...
    #         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
    #                 initial_theta, options);
    #

    # all_theta is a num_labels * (n+1) matrix
    # the i-th row of all_theta corresponds 
    #   to the classifier for label i
    # the j-th element of the i-th row of all_theta correspondes
    #   to the parameter for xj

    # Initialize fitting parameters
    initial_theta = zeros(n+1)

    for i in range(num_labels):
        # Optimize
        maxiter = 50

        #  You should also try different values of lambda
        llambda = 1

        # optimize the neural network parameters
        myargs = (X, (y%10==i).astype(int), llambda)
        myoptions = {'disp': True, 'maxiter':maxiter}
        theta = opt.minimize(lrCostGradientFunction, x0=initial_theta, args=myargs, options=myoptions, method="CG", jac=True)["x"]
        
        # theta is a vector; set it as the i-th row of all_theta
        all_theta [i, :] = theta
       
    # =========================================================================

    return all_theta
