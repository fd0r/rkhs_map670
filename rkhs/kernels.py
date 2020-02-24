import numpy as np


def gaussian_kernel(elt_1, elt_2, sigma=1):
    u = np.sum((elt_1 - elt_2) ** 2)
    return np.exp(- u / (2 * sigma))


def logistic_kernel(elt_1, elt_2):
    u = np.sum((elt_1 - elt_2) ** 2)
    return 1 / (np.exp(u) + 2 + np.exp(-u))
