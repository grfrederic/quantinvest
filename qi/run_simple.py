import tensorflow as tf
import matplotlib.pyplot as plt
from load.load import load_data
from utils.monthify import monthify
from utils.datasets import train_input_fn
import models.gru_simple as gs

tf.enable_eager_execution()


# constants
BATCH_SIZE = 64
SEQ_LENGTH = 10
TRAIN = False


# load and split data
train, valid, tests = load_data()
n_features = train.shape[1]


if TRAIN:
    model = gs.build_model(n_features=n_features,
                           rnn_units=1024,
                           batch_size=BATCH_SIZE)
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss=gs.loss)

    dataset, examples_per_epoch = train_input_fn(train)
    steps_per_epoch = examples_per_epoch // BATCH_SIZE
    history = model.fit(dataset.repeat(),
                        epochs=30,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=[gs.checkpoint_callback])

# LOAD MODEL
model = gs.build_model(n_features=n_features,
                       rnn_units=1024,
                       batch_size=1)
model.load_weights(tf.train.latest_checkpoint(gs.checkpoint_dir))
model.build(tf.TensorShape([1, None]))


# GENERATING
mtests = monthify(tests).astype('float32')

true = mtests[-10:]
pred = gs.generate(model, start=mtests[:-10])

for i in range(n_features):
    plt.plot(true[:, i], color=plt.cm.Set1(i))
    plt.plot(pred[:, i], color=plt.cm.Set1(i),
             linestyle='dashed')

plt.show()
