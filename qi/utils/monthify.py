import numpy as np
from config import MONTH


def monthify(dataset):
    n, n_features = dataset.shape
    n = n // MONTH * MONTH
    month_averages = dataset[:n].reshape((-1, MONTH, n_features, )).mean(axis=1)
    return np.log(month_averages[1:] / month_averages[:-1])
