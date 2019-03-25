from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
import argparse
import time
import datetime
sys.path.append(os.path.abspath('..'))
from python import pruning as pruning
from python.layers import core_layers as core
from examples.cifar10 import cifar10_input
import inference
from mnist_input import MnistData
import tensor_init
import utils
from tensorflow.python.client import device_lib
tf.logging.set_verbosity(tf.logging.INFO)

print(device_lib.list_local_devices())

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
