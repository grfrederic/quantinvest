import tensorflow as tf
import numpy as np
from load.load_all import load_all
from utils.monthify import monthify
from utils.datasets import train_input_fn
import models.gru_disc as gd
from config import MONTH

tf.enable_eager_execution()


# constants
BATCH_SIZE = 100
SEQ_LENGTH = 250 // MONTH
TRAIN = True
DISC = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
N_DISC = len(DISC)


# load and split data
train, tests, monthly_means, monthly_stds = load_all()
n_features = train.shape[1]


print("MONTH:", MONTH)
print("n_features:", n_features)
print("train.shape:", train.shape)

model = gd.build_model(n_features=n_features,
                       n_disc=N_DISC,
                       rnn_units=1024,
                       batch_size=BATCH_SIZE)
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=gd.loss)

dataset, examples_per_epoch = train_input_fn(train,
                                             monthly_means,
                                             monthly_stds,
                                             DISC,
                                             seq_length=SEQ_LENGTH,
                                             batch_size=BATCH_SIZE)


steps_per_epoch = examples_per_epoch // BATCH_SIZE
history = model.fit(dataset.repeat(),
                    epochs=5,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[gd.checkpoint_callback])
