import numpy as np
import tensorflow as tf
from load.load import load_data


train, valid, tests = load_data()
features = train.shape[1]
l = 10
lr = 1e-3
batch_size = 1000

import matplotlib.pyplot as plt
plt.plot(train)
plt.show()

def gen_case(data):
    i = np.random.randint(0, len(data) - l - 1)
    return data[i:i+l], data[i+l+1]


def gen_batch(data):
    xs = []
    ys = []
    for _ in range(batch_size):
        X, Y = gen_case(data)
        xs.append(X)
        ys.append(Y)

    return np.array(xs), np.array(ys)


class Simple(tf.keras.Model):
    def __init__(self):
        super(Simple, self).__init__()
        layers = tf.keras.layers
        self.net = tf.keras.Sequential([
            layers.Reshape(target_shape=(l * features, )),
            layers.Dense(features, activation=tf.nn.relu)
        ])

    def call(self, inputs, training=False, **kwargs):
        return self.net(inputs, training=training)


_, Y = gen_batch(train)
print(f"MSE of single step: {np.sqrt(np.mean(Y * Y))}")


sim = Simple()
inp = tf.placeholder(dtype=tf.float32, shape=(None, l, features))
out = tf.placeholder(dtype=tf.float32, shape=(None, features))
prd = sim(inp)
loss = tf.nn.l2_loss(prd - out)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

print("Learning:")
n_epoch = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, n_epoch+1):
        X, Y = gen_batch(train)
        fd = {inp: X, out: Y}
        curr_loss, _ = sess.run([loss, train_op], feed_dict=fd)
        mse = np.sqrt(2 * curr_loss / batch_size / features)
        if not i % int(n_epoch / 10):
            print(f"Epoch: {i}, MSE: {mse}")
