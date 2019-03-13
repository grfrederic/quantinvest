import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from config import FUNDS, NYSE, UNEMPLOYMENT, WIG20, GDP, IRATES
from utils.monthify import monthify


def load_one(filename):
    data = pd.read_csv(filename)
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by=["date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def load_all(split=0.8):
    funds = load_one(FUNDS)
    nyse = load_one(NYSE)
    unem = load_one(UNEMPLOYMENT)
    wig = load_one(WIG20)
    gdp = load_one(GDP)
    irates = load_one(IRATES)

    START_DATE = max(
        funds["date"].min(),
        nyse["date"].min(),
        unem["date"].min(),
        wig["date"].min(),
        gdp["date"].min(),
        irates["date"].min()
    )

    END_DATE = min(
        funds["date"].max(),
        nyse["date"].max(),
        unem["date"].max(),
        wig["date"].max(),
        gdp["date"].max(),
        irates["date"].max()
    )

    N = (END_DATE - START_DATE).days

    def recalc_date(df):
        df["date"] = list(map(lambda d: float(d.days),
                              df["date"] - START_DATE))
        return df

    funds = recalc_date(funds)
    nyse = recalc_date(nyse)
    unem = recalc_date(unem)
    wig = recalc_date(wig)
    gdp = recalc_date(gdp)
    irates = recalc_date(irates)

    def interpolate(df):
        time = np.array(df["date"])
        vals = np.array(df.loc[:, df.columns != "date"])
        inter_fn = interp1d(time, vals.T)
        time = np.arange(0, N+1)
        vals = inter_fn(time).T
        return vals

    funds_arr = interpolate(funds)
    nyse_arr = interpolate(nyse)
    unem_arr = interpolate(unem)
    wig_arr = interpolate(wig)

    all_arr = np.concatenate((funds_arr.T,
                              nyse_arr.T,
                              unem_arr.T,
                              wig_arr.T)).T

    m_all_arr = monthify(all_arr)
    monthly_means = np.mean(m_all_arr, axis=0)
    m_all_arr = m_all_arr - monthly_means
    monthly_stds = np.std(m_all_arr, axis=0)

    s = int(N * split)
    train = all_arr[:s]
    tests = all_arr[s:]

    return train, tests, monthly_means, monthly_stds
