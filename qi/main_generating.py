import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load.load_all import load_all
from utils.monthify import monthify
import models.gru_disc as gd
from config import MONTH
import time


tf.enable_eager_execution()


# constants
SEQ_LENGTH = 365 // MONTH
TRAIN = True
DISC = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
N_DISC = len(DISC)


# load and split data
_, tests, monthly_means, monthly_stds = load_all()
n_features = tests.shape[1]


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
start = mtests[:SEQ_LENGTH]

# to jest prawdziwy koniec historii
true = mtests[SEQ_LENGTH:2*SEQ_LENGTH]

# trajektorie
t0 = time.time()
batch = np.repeat([start], 1000, axis=0)
preds = gd.generate_many(model, DISC, start=batch, num_generate=SEQ_LENGTH)
preds = np.swapaxes(preds, 0, 1)
print(preds.shape)
t1 = time.time()
print("generating 1000 traj.:", round(t1 - t0, 2), "s")

print(preds[0, :, 0])

exit(0)
to_show = [0, 1, 2]

for i in to_show:
    plt.plot(true[:, i], color=plt.cm.Set1(i))
    for traj in preds:
        plt.plot(traj[:, i], color=plt.cm.Set1(i),
                 alpha = 0.01, linestyle='dashed')

plt.savefig("hmm.png")
