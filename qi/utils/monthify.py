import numpy as np


def monthify(dataset):
    month = 30
    n, n_features = dataset.shape
    n = n // month * month
    month_averages = dataset[:n].reshape((-1, month, n_features, )).mean(axis=1)
    return np.log(month_averages[1:] / month_averages[:-1])
