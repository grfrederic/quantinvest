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
N_GENERATE = int(1e5)
N_BATCH = int(1e4)


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
                       batch_size=N_BATCH)
model.load_weights(tf.train.latest_checkpoint(gd.checkpoint_dir))
model.build(tf.TensorShape([1, None]))


# GENERATING
mtests = monthify(tests).astype('float32')
start = mtests[-3*SEQ_LENGTH:]
batch = np.repeat([start], N_BATCH, axis=0)

future = np.zeros((N_GENERATE, 7))

for st_idx in range(0, N_GENERATE, N_BATCH):
    print("generating traj.:", st_idx, "--", st_idx + N_BATCH)
    t0 = time.time()
    pred = gd.generate_many(model, DISC, start=batch, num_generate=SEQ_LENGTH)
    pred = np.swapaxes(pred, 0, 1)
    pred *= monthly_stds
    pred += monthly_means
    pred = pred[:, :, :7]
    pred = pred.cumsum(axis=1)
    future[st_idx:st_idx+N_BATCH] = pred[:, -1].copy()
    del pred
    t1 = time.time()
    print("time:", round(t1 - t0, 2), "s")


name_end = "." + str(MONTH) + "_" + str(n_features) + ".npy"
np.save("future" + name_end, future)
print("saved future, shape =", future.shape)
