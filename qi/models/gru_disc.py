import tensorflow as tf
import numpy as np
import os


if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools
    rnn = functools.partial(tf.keras.layers.GRU,
                            recurrent_activation='sigmoid')


def build_model(n_features, n_disc, rnn_units, batch_size):
    model = tf.keras.Sequential([
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True,
            dtype='float32',
            batch_input_shape=[batch_size, None, n_features]),
        tf.keras.layers.Dense(n_features * n_disc, dtype='float32'),
        tf.keras.layers.Reshape(target_shape=(-1, n_features, n_disc)),
    ])
    return model


def loss(true, pred):
    return tf.losses.softmax_cross_entropy(true, pred)


checkpoint_dir = './training_checkpoints_gru_disc'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)


temperature = 1.0
def generate(model, disc, start):
    num_generate = 10
    generated = []

    model.reset_states()
    for i in range(num_generate):
        input_eval = tf.expand_dims(start, 0)
        pred = model(input_eval)
        pred = tf.squeeze(pred, 0)
        pred = pred / temperature
        n_batch, n_features, n_disc = pred.shape
        pred_prep = tf.reshape(pred, shape=[-1, n_disc])
        pred_id = tf.random.categorical(pred_prep, num_samples=1)
        pred_id = tf.reshape(pred_id, shape=[-1, n_features])[-1].numpy()
        pred = disc[pred_id]
        input_eval = tf.expand_dims([pred], 0)
        generated.append(pred)

    return np.array(generated)
