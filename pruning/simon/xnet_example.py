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
tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 100
# where model is saved
HOME_DIR = os.path.abspath('../../saved/') + '/'
# inference function. must be compatible with DATA type.
MODEL_FN = inference.lenet300_100
MODEL_NAME = None # gets assigned when program starts
MODEL_DIR = None # gets assigned when program starts

# init dict for tensor init with numpy arrays
# if no initialization desired, set = None
# example: INIT_DICT = tensor_init.emr_big_31()
# must be compatible with MODEL_FN, DATA
INIT_DICT = tensor_init.emr_small_31()

TRAIN_STEPS = 10000
assert TRAIN_STEPS % 100 == 0

# for CIFAR and MNIST, is 10000
TEST_SET_SIZE = 10000
# test this often while training. We don't use validation sets and just test on the test set.
TEST_FREQ = 200
assert TEST_FREQ % 100 == 0

def train_mnist():
    with tf.get_default_graph().as_default():
        input_pipe = MnistData(BATCH_SIZE)
        train_features, train_labels = input_pipe.build_train_data_tensor()

        global_step = 0

        train_op, loss = train_graph(
                train_features, train_labels)

        assign_ops = get_assign_ops()

        checkpoint_hook = basic_session_run_hooks.CheckpointSaverHook(
                MODEL_DIR, save_steps = TEST_FREQ)

        class _LoggerHook(tf.train.SessionRunHook):
            """logs loss and runtime."""

            def begin(self):
                self._step = global_step

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)    # asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step > 0 and self._step % 100 == 0:
                    num_examples_per_step = BATCH_SIZE
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec'
                                  +'; %.3f sec/batch)')
                    tf.logging.info(format_str % (datetime.datetime.now(),
                                    self._step, loss_value, examples_per_sec,
                                    sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=MODEL_DIR,
                    hooks=[_LoggerHook(), checkpoint_hook],
                    save_checkpoint_secs=None,
                    config=tf.ConfigProto(log_device_placement=False)) as sess:

                if INIT_DICT is not None:
                    sess.run(assign_ops)
                    dict_str = ';  '.join(map(lambda scope: scope + '/'
                        + ', '.join(INIT_DICT[scope].keys()) + '',
                        INIT_DICT.keys()))
                    tf.logging.info('Instantiated tensors with assign ops: '
                        + dict_str)

                for i in range(TRAIN_STEPS):
                    sess.run(train_op)

def train_graph(features, labels):
    global_step = tf.train.get_or_create_global_step()
    labels = tf.cast(labels, tf.int64)
    logits = MODEL_FN(features)
    loss = calc_loss(logits, labels)
    predictions = tf.argmax(input=logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(predictions, labels), tf.float32))
    tf.summary.scalar('training accuracy', accuracy)
    train_op = train(loss, global_step)
    return train_op, loss

if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    MODEL_DIR = HOME_DIR + MODEL_NAME
    tf.logging.info(MODEL_DIR)
    train_mnist()
