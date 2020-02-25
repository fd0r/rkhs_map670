import logging

import scipy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model import SGDRegressor, SGDClassifier
from scipy.sparse.linalg import cg
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KKR(BaseEstimator):
    def __init__(self, kernel: str = 'rbf', m: int = 100, lambda_reg: int = 0):
        self.projector = PlainNystrom(kernel=kernel, m=m)
        self.lambda_reg = lambda_reg
        self.coeffs = None

    def fit(self, X: np.ndarray, y: np.array = None, **kwargs):
        n = len(X)
        self.projector.m = n
        k = self.projector.fit_transform(X=X, y=y, **kwargs)
        self.coeffs = np.inv(k + self.lambda_reg * n * np.identity(n)) @ y
        return self

    def predict(self, X):
        projection = self.projector.transform(X=X)
        return projection @ self.coeffs


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

    def fit(self, X: np.ndarray, y: np.array = None, woodbury=False, **kwargs):
        k_nm = self.projector.fit_transform(X=X, y=y, **kwargs)
        k_mm = self.projector.transform(X=self.projector.sample, y=y)
        assert k_mm.shape[0] == k_mm.shape[1] == k_nm.shape[1]
        n = len(X)
        if woodbury:
            raise NotImplementedError
            # self.coeffs = (1 / (self.lambda_reg * n)) * (
            #             np.identity(n) - k_nm @ np.inv(
            #         self.lambda_reg * n * k_mm + k_nm.T @ k_nm) @ k_nm.T) @ y
            # return self
        pseudo_inv = np.linalg.pinv(
            (k_nm.T @ k_nm) + (self.lambda_reg * n * k_mm))
        self.coeffs = pseudo_inv @ k_nm.T @ y
        return self

    def predict(self, X):
        projection = self.projector.transform(X=X)
        return projection @ self.coeffs


# TODO: Implement GD and SGD learning

class GDPlainNystromRegressor:
    def __init__(self):
        raise NotImplementedError


class GDPlainNystromClassifier:
    def __init__(self):
        raise NotImplementedError


class SGDPlainNystromRegressor:
    def __init__(self, kernel: str = 'rbf', m: int = 100, lambda_reg: int = 0,
                 **kwargs):
        self.projector = PlainNystrom(kernel=kernel, m=m)
        self.lambda_reg = lambda_reg
        self.coeffs = None
        self.regressor = SGDRegressor(fit_intercept=False, **kwargs)
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.array = None, **kwargs):
        k_nm = self.projector.fit_transform(X=X, y=y, **kwargs)
        self.regressor.fit(k_nm, y)
        return self

    def predict(self, X):
        projection = self.projector.transform(X=X)
        return self.regressor.predict(projection)


class SGDPlainNystromClassifier:
    def __init__(self, kernel: str = 'rbf', m: int = 100, lambda_reg: int = 0,
                 **kwargs):
        self.projector = PlainNystrom(kernel=kernel, m=m)
        self.lambda_reg = lambda_reg
        self.coeffs = None
        self.classifier = SGDClassifier(fit_intercept=False, **kwargs)
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.array = None, **kwargs):
        k_nm = self.projector.fit_transform(X=X, y=y, **kwargs)
        self.classifier.fit(k_nm, y)
        return self

    def predict(self, X):
        projection = self.projector.transform(X=X)
        return self.classifier.predict(projection)


class FALKON(BaseEstimator):
    def __init__(self, kernel: str = 'rbf', m: int = 100, lambda_reg: float =
    1e-6, max_iter: int = 100, beta_tol=1e-8, verbose=True):
        self.projector = PlainNystrom(kernel=kernel, m=m)
        self.coeffs = None
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.beta_tol = beta_tol
        self.verbose = verbose

    def fit(self, X, y=None, **kwargs):
        n = len(X)
        m = self.projector.m
        epsilon = np.finfo(float).eps
        k_nm = self.projector.fit_transform(X=X, y=y, **kwargs)
        k_mm = self.projector.transform(X=self.projector.sample, y=y)
        T = np.linalg.cholesky(k_mm + epsilon * m * np.eye(m))
        A = np.linalg.cholesky(T @ T.T / m + self.lambda_reg * np.eye(m))

        def knm_times_vector(u, v):
            w = np.zeros(m)
            ms = np.ceil(np.linspace(0, n, np.ceil(n / m) + 1)).astype(int)
            for i in range(int(np.ceil(n / m))):
                kr = self.projector.transform(X[(ms[i] + 1):ms[i + 1]])
                w += kr.T @ (kr @ u + v[(ms[i] + 1):ms[i + 1]])
            return w

        def bhb(u):
            return np.linalg.solve(
                A.T,
                np.linalg.solve(
                    T.T,
                    knm_times_vector(
                        np.linalg.solve(
                            T,
                            np.linalg.solve(A, u)),
                        np.zeros(n)) / n)
                + self.lambda_reg * np.linalg.solve(A, u)
            )

        r = np.linalg.solve(
            A.T,
            np.linalg.solve(
                T.T,
                knm_times_vector(
                    np.zeros(m),
                    y / n)
            )
        )

        def conjgrad(funA, r):
            p = r
            rsold = r.T @ r
            beta = np.zeros(len(r))
            iterator = tqdm(range(self.max_iter), disable=not self.verbose)
            for i in iterator:
                Ap = funA(p)
                a = rsold / (p.T @ Ap)
                beta_old = beta.copy()
                beta += a * p
                r = r - a * Ap
                rsnew = r.T @ r
                p = r + (rsnew / rsold) * p
                imrpovement = np.sum((beta - beta_old) ** 2)
                iterator.set_postfix(improvement=imrpovement)
                if imrpovement < self.beta_tol:
                    return beta
                rsold = rsnew
            logger.warning(
                "Conjugate Gradient did not converge in {} "
                "steps".format(self.max_iter))
            return beta

        self.coeffs = np.linalg.solve(
            T,
            np.linalg.solve(
                A,
                conjgrad(bhb, r)
            )
        )
        return self
    
    def predict(self, X):
        projection = self.projector.transform(X=X)
        return projection @ self.coeffs

    def predict(self, X):
        if self.coeffs is None:
            raise Exception("Regressor must be fitted")
        projection = self.projector.transform(X=X)
        return projection @ self.coeffs


# TODO: Implement hyper-parameter search with incremental method

class ALSNystrom:
    # TODO: Implement this using; https://github.com/LCSL/bless
    def __init__(self):
        raise NotImplementedError(
            "Must find scalable way to compute leverage scores")
