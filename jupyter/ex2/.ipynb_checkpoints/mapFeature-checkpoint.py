from numpy import ones, power

def mapFeature(X1, X2):
    """
    MAPFEATURE Feature mapping function to polynomial features

    MAPFEATURE(X1, X2) maps the two input features to quadratic features used in the regularization exercise.
    Returns a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    Inputs X1, X2 must be the same size
    """

    degree = 6
    out = ones(( X1.shape[0], sum(range(degree + 2)) )) # could also use ((degree+1) * (degree+2)) / 2 instead of sum
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i+1):
            out[:,curr_column] = power(X1,i-j) * power(X2,j)
            curr_column += 1

    return out