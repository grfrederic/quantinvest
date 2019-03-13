import tensorflow as tf
from utils.monthify import monthify
from config import MONTH


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def train_input_fn(data,
                   seq_length=10,
                   batch_size=64):

    examples_per_epoch = 0
    datasets = []
    for i in range(MONTH):
        mdata = monthify(data[i:]).astype('float32')
        dataset = tf.data.Dataset.from_tensor_slices(mdata)
        dataset = dataset.batch(seq_length + 1,
                                drop_remainder=True)
        dataset = dataset.map(split_input_target)
        datasets.append(dataset)
        examples_per_epoch += len(mdata)

    dataset = datasets[0]
    for d in datasets[1:]:
        dataset = dataset.concatenate(d)

    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, examples_per_epoch
