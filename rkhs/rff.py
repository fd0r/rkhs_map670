import numpy as np

class RFF(object):
    def __init__(self, D=100, kernel="gaussian", gamma="scale", method=0):
        self.D = D
        self.kernel = kernel
        self.gamma = gamma
        self.method = method

    def fit(self, X, y=None):
        d = X.shape[1]
        if self.kernel == "gaussian":
            mu = np.zeros(d)
            if self.gamma == "scale":
                sigma = (1/(d*X.var()))*np.identity(d)
            elif self.gamma == "auto":
                sigma = (1/d)*np.identity(d)
            elif type(self.gamma) == float:
                sigma = self.gamma*np.identity(d)
            else:
                raise TypeError("gamma must be either 'scale', 'auto', or a float.")
            self.w = np.random.multivariate_normal(mu, sigma, self.D)
            if self.method == 0:
                pass
            elif self.method == 1:
                self.b = np.random.uniform(0, 2*np.pi, self.D)
            else:
                raise TypeError("method must be either 0 or 1.")
        return self

    def transform(self, X):
        if self.kernel == "gaussian":
            if self.method == 0:
                return np.sqrt(1/self.D) * np.concatenate((np.cos(np.dot(X, self.w.T)), np.sin(np.dot(X, self.w.T))), axis=1)
            if self.method == 1:
                return np.sqrt(2/self.D) * np.cos(np.dot(X, self.w.T) + self.b)
            else:
                raise TypeError("method must be either 0 or 1.")
