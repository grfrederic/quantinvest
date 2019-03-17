import os.path


# root = location of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')


# the different datasets
FUNDS = os.path.join(DATA_DIR, 'qi_fundusze.csv')
NYSE = os.path.join(DATA_DIR, 'nyse.csv')
UNEMPLOYMENT = os.path.join(DATA_DIR, 'unemployment.csv')
WIG20 = os.path.join(DATA_DIR, 'wig20.csv')
GDP = os.path.join(DATA_DIR, 'gdp.csv')
IRATES = os.path.join(DATA_DIR, 'irates.csv')
USD = os.path.join(DATA_DIR, 'usd.csv')
USDPLN = os.path.join(DATA_DIR, 'usdpln.csv')
VIX = os.path.join(DATA_DIR, 'vix.csv')


# constants
MONTH = 21
