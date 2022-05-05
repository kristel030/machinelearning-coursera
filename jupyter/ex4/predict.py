import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X_data):
    """
    PREDICT Predict the label of an input given a trained neural network
    
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    
    # Useful values
    m = X_data.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The np.argmax function might come in useful 
    #       to determine the index of highest value on a particular row or column. 
    #       np.argmax(A, axis=1) determines the index of the highest value on each row of A
    #

    # Add ones to the X data matrix, and compute A2 with Theta1
    X = np.append( np.ones((m,1)), X_data, axis=1)
    A2 = sigmoid(X @ Theta1.T)

    # Add ones to the A2 data matrix, and compute A3 with Theta2
    A2 = np.append( np.ones((m,1)), A2, axis=1)
    A3 = sigmoid(A2 @ Theta2.T)
    
    # A3 is a m x num_labels matrix
    # - each row represent a single training sample
    # - each column represets a class
    # - each cell on a row represents the chance that the training sample belongs to class c
    # - by determining the column index of the highest value on each row, 
    #   we know to which class the machine learning algorithm has assigned this training sample
    p = np.argmax(A3, axis=1)
    p = p+1

    # =========================================================================
    
    return p
