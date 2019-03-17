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
start = mtests

t0 = time.time()
batch = np.repeat([start], 1000, axis=0)
pred = gd.generate_many(model, DISC, start=batch, num_generate=SEQ_LENGTH)
pred = np.swapaxes(pred, 0, 1)
t1 = time.time()
print("generating 1000 traj.:", round(t1 - t0, 2), "s")

pred *= monthly_stds
pred += monthly_means

name_end = "." + str(MONTH) + "_" + str(n_features) + ".npy"

np.save("future" + name_end, pred)
print("saved preds, shape =", pred.shape)
