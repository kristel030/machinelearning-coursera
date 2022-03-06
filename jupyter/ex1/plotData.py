import matplotlib.pyplot as plt

def plotData(X, y):
    #plt.scatter(data[:, 0], data[:, 1])
    plt.title("Foodtruck profit")
    plt.xlabel("City population in 10,000s")
    plt.ylabel("Profit in $10,000s")

    plt.scatter(X, y, marker="x", color="red")
    