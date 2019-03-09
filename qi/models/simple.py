import tensorflow as tf


class Simple(tf.keras.Model):
    def __init__(self, n_seq, n_val):
        super(Simple, self).__init__()
        layers = tf.keras.layers
        self.net = tf.keras.Sequential([
            layers.Reshape(target_shape=(n_seq * n_val, )),
            layers.Dense(2 * n_val, activation=tf.nn.relu),
            layers.Reshape(target_shape=(2, n_val, ))
        ])

    def call(self, inputs, training=False, **kwargs):
        return self.net(inputs, training=training)


def model_fn(features, labels, mode):
    n_batch, n_seq, n_val = features.shape
    simple = Simple(n_seq, n_val)

    if mode == tf.estimator.ModeKeys.PREDICT:
        dist_all = simple(features, training=False)
        dist_mean = dist_all[:, 0]
        dist_log_var = dist_all[:, 1]
        dist_std = tf.exp(0.5 * dist_log_var)
        epsilon = tf.random_normal(shape=(n_batch, n_val))
        dist_out = dist_mean + dist_std * epsilon
        predictions = {'logd': dist_out}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    training = mode == tf.estimator.ModeKeys.TRAIN
    dist_all = simple(features, training=training)
    dist_mean = dist_all[:, 0]
    dist_log_var = dist_all[:, 1]
    dist_var = tf.exp(dist_log_var)

    loss = tf.reduce_mean(- dist_log_var - (labels - dist_mean)**2 / dist_var / 2)

    # EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    # TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

