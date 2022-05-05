import numpy as np
from printDim import printDim
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def nnCostFunction2(nn_params, input_layer_size, hidden_layer_size, num_labels, X_data, y, llambda):
    """
    NNCOSTFUNCTION Implements the neural network cost function for a two layer neural network which performs classification

    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels,  X, y, lambda) computes the cost and gradient of the neural network. 
    The  parameters for the neural network are "unrolled" into the vector nn_params and need to be converted back into the weight matrices. 

    The returned parameter grad should be a "unrolled" vector of the partial derivatives of the neural network.
    """

    # Reshape nn_params back into the parameters Theta1 and Theta2, 
    # the weight matrices for our 2 layer neural network
    Theta1_1d = nn_params[: hidden_layer_size * (input_layer_size + 1)]
    Theta1 = np.reshape(Theta1_1d, (hidden_layer_size, input_layer_size + 1))
    
    Theta2_1d = nn_params[hidden_layer_size * (input_layer_size + 1):]
    Theta2 = np.reshape(Theta2_1d, (num_labels, hidden_layer_size + 1))
                          
    # printDim(X_data, 'X_data')
    # printDim(y, 'y')
    # printDim(Theta1, 'Theta1')
    # printDim(Theta2, 'Theta2')

    # Setup some useful variables
    m = X_data.shape[0]
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the variable J. 
    #         After implementing Part 1, you can verify that your cost function computation is correct
    #         by verifying the cost computed in ex4.ipynb

    J_base = 0
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)

    # printDim(Delta1, 'Delta1')
    # printDim(Delta2, 'Delta2')

    # remove bias unit weights from the Theta matrices (this is the first column)
    Theta1_ = Theta1[:, 1:]
    # printDim(Theta1_, 'Theta1_')
    Theta2_ = Theta2[:, 1:]
    # printDim(Theta2_, 'Theta2_')

    # iterate over the training examples
    for i in range(m):
        # FORWARD PROPAGATION: calculate a1 (=x), a2 and a3 (=h) for every example i
        
        # determine a1_i
        x_i = X_data[i, :]
        a1_i = x_i
        # if (i == 0):
            # printDim(a1_i, 'a1_i')

        # calculate a2_i
        a1_i = np.append(1, a1_i) # Add a 1 for the bias unit
        z2_i = a1_i @ Theta1.T
        a2_i = sigmoid(z2_i)
        # if (i == 0):
            # printDim(a2_i, 'a2_i')

        # calculate a3_i (= h_i)
        a2_i = np.append(1, a2_i) # % Add a 1 for the bias unit
        z3_i = a2_i @ Theta2.T
        a3_i = sigmoid(z3_i)
        h_i = a3_i
        # if (i==0):
            # printDim(h_i, 'h_i')
            # print('h_i: ' + str(h_i) )
        
        # determine y_i, a vector representation of y[i]
        # --> y_i is a vector with length 'num_labels'
        # --> if y[i] = 3, then the 3rd element (index = 2) is 1, while the other values are 0
        # --> in other words: y[i] = 3 leads to y_i = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        y_i = np.zeros(num_labels)
        y_i[y[i]-1] = 1
        # if (i==0):
            # printDim(y_i, 'y_i')
            # print('y_i: ' + str(y_i) )
            # print('y[i]: ' + str(y[i]) )

        # calculate the costs for this training example and add it to the total costs (J_base)
        # J_i = (-y_i*log(h_i)' - (1-y_i) * log(1-h_i)');
        J_i = (-y_i @ np.log(h_i).T - (1-y_i) @ np.log(1-h_i).T)
        J_base = J_base + J_i
        
        # BACK PROPAGATION
        d3_i = a3_i - y_i
        # printDim(d3_i, 'd3_i')
        d2_i = (Theta2_.T @ d3_i).T * sigmoidGradient(z2_i)
        # printDim(d2_i, 'd2_i')
        
#         printDim(Delta2, 'Delta2')
#         printDim(a2_i, 'a2_i')
#         print('...+= np.outer(d3_i, a2_i)')
        Delta2 += np.outer(d3_i, a2_i)
    
        # if (i==0):
        #     print('...Delta1 += np.outer(d2_i, a1_i)')
        #     printDim(Delta1, 'Delta1')
        #     printDim(d2_i, 'd2_i')
        #     printDim(a1_i, 'a1_i')
        Delta1 += np.outer(d2_i, a1_i)
    
    # cost implementation *without* regularization
    J = J_base / m
    
    # cost implementation *with* regularization  
    J += (llambda / (2*m)) * ( np.sum(Theta1_**2) + np.sum(Theta2_**2) ) 
    
    # gradient implementation *without* regularization
    # Theta1_grad = Delta1 / m 
    # Theta2_grad = Delta2 / m
    
     # gradient implementation *with* regularization
    Theta1_grad = (Delta1 + llambda * Theta1) / m 
    Theta1_grad[:,0] = Delta1[:,0] / m 
    Theta2_grad = (Delta2 + llambda * Theta2) / m
    Theta2_grad[:,0] = Delta2[:,0] / m 
    
    # =================================
    
    # return (J, np.append(Theta1_grad.flatten(), Theta2_grad.flatten()) )
    return np.append(Theta1_grad.flatten(), Theta2_grad.flatten()) 