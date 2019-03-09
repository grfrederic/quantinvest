import numpy as np
import tensorflow as tf
from load.load import load_data


train, valid, tests = load_data()

from utils.datasets import input_fn
from models.simple import model_fn

simple_classifier = tf.estimator.Estimator(model_fn=model_fn)
simple_classifier.train(input_fn=lambda: input_fn(train))
simple_classifier.evaluate(input_fn=lambda: input_fn(valid))