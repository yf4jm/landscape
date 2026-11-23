import numpy as np
from .optimization.gradient import grad
from .optimization.loss import loss
class LinearRegression():
    def __init__(self,lr=0.2,epochs=300, debug=False):
        self.lr, self.epochs = lr, epochs
        self.theta = None
        self.debug = debug
    def _add_intercept(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)          # (n,) -> (n, 1)
        return np.column_stack([np.ones(len(X)), X])
    def model(self, X):
        return self._add_intercept(X) @ self.theta
    def fit(self, X_train, y_train):
        Xb = np.column_stack([np.ones(len(X_train)), X_train]) # used to eXtract how many varibles we have using Xb.shape[1]
        self.theta = np.zeros(Xb.shape[1])
        self.loss_history = []
        for i in range(self.epochs):
            L = loss(Xb, y_train, self.theta)
            if self.debug:
                print(f"iter {i:3d}  theta={self.theta}  loss={L:.6f}")
            gradient = grad(y_train,Xb,self.theta)
            self.loss_history.append(L)
            self.theta -= self.lr * gradient
        return self

    def predict(self, X):
        return self.model(X)

# ref 1 https://numpy.org/doc/2.1/reference/routines.linalg.html