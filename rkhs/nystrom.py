import logging
from typing import Callable

from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np

logger = logging.getLogger(__name__)


class PlainNystrom:
    def __init__(self, kernel: str = 'rbf', m: int = 100):
        self.m = m
        self.kernel = kernel
        self.kernels = None

    def fit(self, X: np.ndarray, y: np.array = None, **kwargs):
        c = X[:]
        np.random.shuffle(c)
        sample = c[:self.m]
        self.kernels = lambda elt: pairwise_kernels(
            elt, sample, metric=self.kernel, **kwargs)
        return self

    def transform(self, X: np.ndarray, y: np.array = None):
        return self.kernels(X)

    def fit_transform(self, X: np.ndarray, y: np.array = None):
        self.fit(X, y)
        return self.transform(X, y)
