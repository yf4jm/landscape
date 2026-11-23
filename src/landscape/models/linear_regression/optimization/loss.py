import numpy as np
def loss(Xb, y, theta):
    r = Xb @ theta - y
    return np.mean(r ** 2)