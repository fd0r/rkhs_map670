import logging

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels

logger = logging.getLogger(__name__)


class PlainNystrom:
    def __init__(self, kernel: str = 'rbf', m: int = 100):
        self.m = m
        self.kernel = kernel
        self.sample = None

    def fit(self, X: np.ndarray, y: np.array = None):
        self.sample = X[
            np.random.choice(len(X), min(self.m, len(X)), replace=False)]
        return self

    def transform(self, X: np.ndarray, y: np.array = None, **kwargs):
        return pairwise_kernels(
            X, self.sample, metric=self.kernel, **kwargs)

    def fit_transform(self, X: np.ndarray, y: np.array = None, **kwargs):
        self.fit(X, y)
        return self.transform(X, y, **kwargs)


class PlainNystromRegressor(BaseEstimator):
    def __init__(self, kernel: str = 'rbf', m: int = 100, lambda_reg: int = 0):
        self.projector = PlainNystrom(kernel=kernel, m=m)
        self.lambda_reg = lambda_reg
        self.coeffs = None

    def fit(self, X: np.ndarray, y: np.array = None, **kwargs):
        k_nm = self.projector.fit_transform(X=X, y=y, **kwargs)
        k_mm = self.projector.transform(X=self.projector.sample, y=y)
        assert k_mm.shape[0] == k_mm.shape[1] == k_nm.shape[1]
        n = len(X)
        pseudo_inv = np.linalg.pinv(
            (k_nm.T @ k_nm) + (self.lambda_reg * n * k_mm))
        self.coeffs = pseudo_inv @ k_nm.T @ y
        return self

    def predict(self, X):
        projection = self.projector.transform(X=X)
        return projection @ self.coeffs


class ALSNystrom:
    # TODO: Implement this using; https://github.com/LCSL/bless
    def __init__(self):
        raise NotImplementedError(
            "Must find scalable way to compute leverage scores")
