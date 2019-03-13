import tensorflow as tf
import numpy as np
import os


if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(tf.keras.layers.GRU,
                            recurrent_activation='sigmoid')


def build_model(n_features, rnn_units, batch_size):
    model = tf.keras.Sequential([
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True,
            dtype='float32',
            batch_input_shape=[batch_size, None, n_features]),
        tf.keras.layers.Dense(n_features, dtype='float32')
    ])
    return model


def loss(true, pred):
    return tf.nn.l2_loss(true - pred)


checkpoint_dir = './training_checkpoints_gru_simple'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)


def generate(model, start):
    num_generate = 10
    generated = []

    model.reset_states()
    for i in range(num_generate):
        input_eval = tf.expand_dims(start, 0)
        pred = model(input_eval)
        pred = tf.squeeze(pred, 0)
        input_eval = tf.expand_dims([pred], 0)
        generated.append(pred[-1].numpy())

    return np.array(generated)
