import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y, title, xLabel, yLabel):
    # indices of admitted and not-admitted examples
    indices_1 = np.where(y==1)
    indices_0 = np.where(y==0)
    
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    p1 = plt.scatter(X[indices_1][:,0],X[indices_1][:,1], marker="+", color="black")
    p2 = plt.scatter(X[indices_0][:,0], X[indices_0][:,1], marker="o", color="yellow")
    
    plt.legend((p1, p2), (xLabel, yLabel), numpoints=1, handlelength=0)

    return plt

#     p1 = plt.plot(X[indices_1][:,0],X[indices_1][:,1], marker="+", color="black")[0]
#     p2 = plt.plot(X[indices_0][:,0], X[indices_0][:,1], marker="o", color="yellow")[0]
    
#     return plt, p1, p2
    