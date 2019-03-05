import os.path

# root = location of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')


# the different datasets
FUNDS = os.path.join(DATA_DIR, 'fundusze.csv')
