import tensorflow as tf
import numpy as np


def _generator(dataset, seq_len=10, batch_size=200, batch_per_epoch=10):
    month = 30
    n, n_features = dataset.shape
    true_seq_len = month * (seq_len + 2)

    def gen():
        for _ in range(batch_per_epoch):
            batch = np.zeros((batch_size, seq_len+1, n_features, ))
            for i in range(batch_size):
                st = np.random.randint(0, n - true_seq_len)
                month_averages = dataset[st:st+true_seq_len].reshape((-1, month, n_features, )).mean(axis=1)
                batch[i] = np.log(month_averages[1:] / month_averages[:-1])

            features = batch[:, :-1]
            labels = batch[:, -1]
            yield (features, labels)

    return gen


def input_fn(dataset, seq_len=10, batch_size=200, batch_per_epoch=100, n_epoch=10):
    _, n_features = dataset.shape
    gen = _generator(dataset, seq_len, batch_size, batch_per_epoch)

    shape_x = tf.TensorShape([batch_size, seq_len, n_features])
    shape_y = tf.TensorShape([batch_size, n_features])
    dataset = tf.data.Dataset.from_generator(
        gen, (tf.float32, tf.float32), (shape_x, shape_y)
    )

    return dataset.repeat(count=n_epoch)
