#! python3

from . import vgg
import tensorflow as tf

from ..data import cifar10

print("Training model: {}".format(model.NAME))

n_input = tf.placeholder(tf.float32, shape=cifar10.get_shape(), name="input")
n_output = vgg.build(n_input)