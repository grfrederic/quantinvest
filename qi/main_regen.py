import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from load.load_all import load_all
from utils.monthify import monthify
import models.gru_disc as gd
from config import MONTH
import time


tf.enable_eager_execution()


# constants
SEQ_LENGTH = 250 // MONTH
DISC = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
N_DISC = len(DISC)


# load and split data
_, tests, monthly_means, monthly_stds = load_all()
n_features = tests.shape[1]


print("MONTH:", MONTH)
print("n_tests:", len(tests))
print("n_features:", n_features)


# LOAD MODEL
model = gd.build_model(n_features=n_features,
                       n_disc=N_DISC,
                       rnn_units=1024,
                       batch_size=1000)
model.load_weights(tf.train.latest_checkpoint(gd.checkpoint_dir))
model.build(tf.TensorShape([1, None]))


# GENERATING
mtests = monthify(tests).astype('float32')
mtests -= monthly_means
mtests /= monthly_stds

starts = []
trues = []
for st_idx in range(0, len(mtests) - 2*SEQ_LENGTH, 21 // MONTH):
    start = mtests[st_idx:st_idx+SEQ_LENGTH].copy()
    # to jest prawdziwy koniec historii
    true = mtests[st_idx+SEQ_LENGTH:st_idx+2*SEQ_LENGTH].copy()

    start *= monthly_stds
    start += monthly_means

    true *= monthly_stds
    true += monthly_means

    starts.append(start)
    trues.append(true)

name_end = "." + str(MONTH) + "_" + str(n_features) + ".npy"

starts = np.array(starts)
np.save("start" + name_end, starts)
print("saved starts, shape =", starts.shape)
del starts

trues = np.array(trues)
np.save("true" + name_end, trues)
print("saved trues, shape =", trues.shape)
del trues
