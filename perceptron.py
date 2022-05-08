import numpy as np
import math


class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target-self.predict_obs(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict_obs(self, X):
        prob = 1/(1+math.exp(-self.net_input(X)))
        if prob <= 0.333 and prob >= 0:
            result = 0
        elif prob > 0.333 and prob <= 0.666:
            result = 1
        elif prob <= 1 and prob > 0.666:
            result = 2
        return result

    def predict(self, X):
        results = []
        for xi in X:
            results.append(self.predict_obs(xi))
        return np.array(results)