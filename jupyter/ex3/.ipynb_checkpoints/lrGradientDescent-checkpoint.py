from numpy import zeros
from sigmoid import sigmoid

def lrGradientDescent(theta, X, y, lambda_t):
    # print('****************hello')
    m = len(y)
    grad = zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    # print('grad: ' + grad)
    return grad