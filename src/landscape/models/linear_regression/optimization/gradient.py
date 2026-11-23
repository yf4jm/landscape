def grad(y, Xb, theta):
    return (2 / len(y)) * Xb.T @ (Xb @ theta - y)

