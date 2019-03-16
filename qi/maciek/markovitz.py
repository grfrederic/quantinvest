#!/usr/bin/env python
import pandas as pd
import numpy as np
import pypfopt
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt

def markovitz(data, target_return):
  '''
  Calculate markovitz portfolio for given historical/simulated returns
  for specified target rate of return  
  Columns in data represent assets, rows represent simulated returns
  '''
  ef = EfficientFrontier(data.mean(), data.cov(), weight_bounds=(-1, 1))
  proportions = ef.efficient_return(target_return)
  mean, std, sharpe = ef.portfolio_performance(risk_free_rate=0, verbose=False)
  return proportions, mean, std


def plot_frontier(data, target_returns, labels):
  '''
  calculates markovitz portfolios for given target returns,
  plots and returns them
  '''
  
  data = pd.DataFrame(data=data, columns=labels)
  
  for col in data:
    m, s = data[col].mean(), data[col].std() 
    plt.scatter(s, m, color='blue', alpha=0.65)
    plt.annotate(col, (s, m), fontsize='large')
  
  # optimize portfolios
  results = list(markovitz(data, x) for x in target_returns)

  # plot frontier
  means = [z[1] for z in results]
  stds = [z[2] for z in results]
  plt.plot(stds, means)
  plt.xlabel('Standard deviation')
  plt.ylabel('Expected yield')
  plt.grid()
  
  return results
    

