import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load.load_all import load_all
from utils.monthify import monthify
from utils.datasets import train_input_fn
import models.gru_disc as gd

tf.enable_eager_execution()


# constants
BATCH_SIZE = 64
SEQ_LENGTH = 10
TRAIN = False
DISC = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
N_DISC = len(DISC)


# load and split data
train, tests, monthly_means, monthly_stds = load_all()
n_features = train.shape[1]


if TRAIN:
    model = gd.build_model(n_features=n_features,
                           n_disc=N_DISC,
                           rnn_units=1024,
                           batch_size=BATCH_SIZE)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=gd.loss)

    dataset, examples_per_epoch = train_input_fn(train,
                                                 monthly_means,
                                                 monthly_stds,
                                                 DISC)

    steps_per_epoch = examples_per_epoch // BATCH_SIZE
    history = model.fit(dataset.repeat(),
                        epochs=3,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[gd.checkpoint_callback])


# LOAD MODEL
model = gd.build_model(n_features=n_features,
                       n_disc=N_DISC,
                       rnn_units=1024,
                       batch_size=1)
model.load_weights(tf.train.latest_checkpoint(gd.checkpoint_dir))
model.build(tf.TensorShape([1, None]))


# GENERATING
mtests = monthify(tests).astype('float32')
mtests -= monthly_means
mtests /= monthly_stds

true = mtests[-10:]
pred = gd.generate(model, DISC, start=mtests[:-10])

for i in range(n_features):
    plt.plot(true[:, i], color=plt.cm.Set1(i))
    plt.plot(pred[:, i], color=plt.cm.Set1(i),
             linestyle='dashed')

plt.show()
