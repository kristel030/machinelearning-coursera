from lrCostFunction import lrCostFunction
from lrGradientDescent import lrGradientDescent

def lrCostGradientFunction(theta, X, y, llambda):
    
    return (lrCostFunction(theta, X, y, llambda), lrGradientDescent(theta, X, y, llambda))