import pandas as pd
import numpy as np
from config import FUNDS


# load data
def load_data(split=(8, 1, 1)):
    data = pd.read_csv(FUNDS)
    data.sort_values("time", inplace=True)

    data = np.array(data)
    diff = data[1:] - data[:-1]            # difference table
    dxdt = (diff[:, 1:].T / diff[:, 0]).T  # dx / dt

    norm = dxdt - np.mean(dxdt, axis=0)
    norm = norm / np.sqrt(np.mean(norm*norm, axis=0))

    s = sum(split)
    n = len(data)

    i = int(n * split[0] / s)
    j = int(n * split[1] / s)

    train = norm[:i]
    valid = norm[i:j]
    tests = norm[j:]

    return train, valid, tests
