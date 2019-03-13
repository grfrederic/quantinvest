import tensorflow as tf
import numpy as np
from utils.monthify import monthify
from config import MONTH


def splitter_fn(disc):
    def splitter(chunk):
        input_data = chunk[:-1]
        target_data = chunk[1:]
        target_data = tf.expand_dims(target_data, axis=-1)
        target_data = tf.argmin(tf.abs(target_data - disc), axis=-1)
        target_data = tf.one_hot(target_data, len(disc), axis=-1)
        return input_data, target_data
    return splitter


def train_input_fn(data, monthly_means, monthly_stds, disc,
                   seq_length=10,
                   batch_size=64):

    examples_per_epoch = 0
    datasets = []
    for i in range(MONTH):
        mdata = monthify(data[i:]).astype('float32')
        mdata -= monthly_means
        mdata /= monthly_stds
        dataset = tf.data.Dataset.from_tensor_slices(mdata)
        dataset = dataset.batch(seq_length + 1,
                                drop_remainder=True)
        dataset = dataset.map(splitter_fn(disc))
        datasets.append(dataset)
        examples_per_epoch += len(mdata)

    dataset = datasets[0]
    for d in datasets[1:]:
        dataset = dataset.concatenate(d)

    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, examples_per_epoch
