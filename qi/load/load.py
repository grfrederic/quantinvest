import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from config import FUNDS


# load data
def load_data(split=(8, 1, 1)):
    data = pd.read_csv(FUNDS)
    data.sort_values(by=["time"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    time = np.array(data["time"])
    vals = np.array(data.loc[:, data.columns != "time"])

    # interpolation
    inter_fn = interp1d(time, vals.T)

    time = np.arange(np.min(time), np.max(time))
    vals = inter_fn(time).T

    s = sum(split)
    n = len(vals)

    i = int(n * split[0] / s)
    j = i + int(n * split[1] / s)

    train = vals[:i]
    valid = vals[i:j]
    tests = vals[j:]

    return train, valid, tests
