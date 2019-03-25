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
# to use MNIST data, use DATA = 'MNIST'
# to use CIFAR data, use DATA = 'CIFAR'
DATA = 'MNIST'
# where model is saved
HOME_DIR = os.path.abspath('../../saved/') + '/'
# inference function. must be compatible with DATA type.
MODEL_FN = inference.lenet5
MODEL_NAME = None # gets assigned when program starts
MODEL_DIR = None # gets assigned when program starts

# init dict for tensor init with numpy arrays
# if no initialization desired, set = None
# example: INIT_DICT = tensor_init.emr_big_31()
# must be compatible with MODEL_FN, DATA
INIT_DICT = None

TRAIN_STEPS = 10000
assert TRAIN_STEPS % 100 == 0

# for CIFAR and MNIST, is 10000
TEST_SET_SIZE = 10000
# test this often while training. We don't use validation sets and just test on the test set.
TEST_FREQ = 200
assert TEST_FREQ % 100 == 0

"""
For restoring. the vars names come from the old model, as do the shapes. 
"""
RESTORE = False
RESTORE_VARS = ['conv1/mask', 'conv2/mask', 'dense1/mask', 'logits/mask']
RESTORE_SHAPES = [(5, 5, 1, 20), (5, 5, 20, 50), (1800, 384), (384, 10)]
RESTORE_DIR = HOME_DIR + '../aug22/cifar_ppc_rt/0.5/'

# pruning params
PRUNING_ON = False
BEGIN_PRUNING_STEP = 3000
END_PRUNING_STEP = 10000
INITIAL_SPARSITY = 0.0
TARGET_SPARSITY = 0.96
SPARSITY_FUNCTION_BEGIN_STEP = BEGIN_PRUNING_STEP
SPARSITY_FUNCTION_END_STEP = END_PRUNING_STEP
PRUNING_FREQUENCY = 200
assert PRUNING_FREQUENCY % 100 == 0

# if set to None, prune checkpointing will not be done
# otherwise, will save a version of the model at that pruning level.
# should be a list of sparsity levels i.e. [0.5, 0.75]
# PRUNING_CKPTS = [0.5, 0.75, 0.9, 0.95]
PRUNING_CKPTS = None

PRUNING_HPARAMS = 'pruning_on='+str(PRUNING_ON) + ',' \
    + 'begin_pruning_step=' + str(BEGIN_PRUNING_STEP) + ',' \
    + 'end_pruning_step=' + str(END_PRUNING_STEP) + ',' \
    + 'initial_sparsity=' + str(INITIAL_SPARSITY) + ',' \
    + 'target_sparsity=' + str(TARGET_SPARSITY) + ',' \
    + 'sparsity_function_begin_step=' + str(SPARSITY_FUNCTION_BEGIN_STEP) + ',' \
    + 'sparsity_function_end_step=' + str(SPARSITY_FUNCTION_END_STEP) + ',' \
    + 'pruning_frequency=' + str(PRUNING_FREQUENCY) \

def get_assign_ops():
    assign_ops = []
    if not INIT_DICT:
        return assign_ops

    for scope in INIT_DICT:
        var_dict = INIT_DICT[scope]
        with tf.variable_scope(scope, reuse=True):
            for var in INIT_DICT[scope]:
                assign_op = tf.assign(tf.get_variable(var), var_dict[var])
                assign_ops.append(assign_op)

    return assign_ops


def train_mnist():
    """
    The main function which trains. Does basically everything in one function.
    """
    with tf.get_default_graph().as_default():
        if DATA is 'MNIST':
            input_pipe = MnistData(BATCH_SIZE)
            train_features, train_labels = input_pipe.build_train_data_tensor()
        elif DATA is 'CIFAR10':
            train_features, train_labels = cifar10_input.distorted_inputs(BATCH_SIZE)
        else:
            raise ValueError('DATA value not supported: ' + DATA)

        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
        if checkpoint is None:
            tf.logging.info('No checkpoint found; training from scratch')
            global_step = 0
        else:
            # Assuming model_checkpoint_path looks something like:
            #       /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            # needs to be before graph setup
            global_step = float(checkpoint.model_checkpoint_path
                                .split('/')[-1].split('-')[-1])
            tf.logging.info('Continuing training from step: ' + str(global_step))


        train_op, mask_update_op, pruning_obj, loss = train_graph(
            train_features, train_labels)

        assign_ops = get_assign_ops()

        checkpoint_hook = basic_session_run_hooks.CheckpointSaverHook(
            MODEL_DIR, save_steps = TEST_FREQ)

        # needs to be after graph setup
        if checkpoint is None:
            if RESTORE:
                var_list = []
                for var_str, shape in zip(RESTORE_VARS, RESTORE_SHAPES):
                    i = var_str.index('/')
                    scope = var_str[:i]
                    with tf.variable_scope(scope, reuse=True):
                        var = tf.get_variable(var_str[i+1:], shape=shape)
                        var_list.append(var)

                saver = tf.train.Saver(var_list)
        else:
#            cifar_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#            cifar_vars = cifar_vars[:-1]
#            for var in cifar_vars:
#                tf.logging.info(str(var))
#            saver2 = tf.train.Saver(cifar_vars)
            saver2 = tf.train.Saver()

        class _LoggerHook(tf.train.SessionRunHook):
            """logs loss and runtime."""

            def begin(self):
                self._step = -0.5 + global_step

            def before_run(self, run_context):
                # this is a hack so that it correctly counts
                # (trainop, mask_update_op) as one step
                self._step += 0.5
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
            if checkpoint is not None:
                saver2.restore(sess, checkpoint.model_checkpoint_path)
            elif RESTORE:
                ckpt_path = tf.train.latest_checkpoint(RESTORE_DIR)
                if not ckpt_path:
                    raise ValueError('Restore dir cant find checkpoint fles: '
                        + RESTORE_DIR)
                saver.restore(sess, ckpt_path)

            if INIT_DICT is not None:
                sess.run(assign_ops)
                dict_str = ';  '.join(map(lambda scope: scope + '/'
                    + ', '.join(INIT_DICT[scope].keys()) + '',
                    INIT_DICT.keys()))
                tf.logging.info('Instantiated tensors with assign ops: '
                    + dict_str)

            if PRUNING_CKPTS:
                pruning_dir = MODEL_DIR + '/../' + MODEL_NAME + '_prune_ckpts/'
                if not os.path.isdir(pruning_dir):
                    os.mkdir(pruning_dir)

                j = 0
                max_j = len(PRUNING_CKPTS) 

            for i in range(TRAIN_STEPS):
                sess.run(train_op)
                sess.run(mask_update_op)

                if (PRUNING_CKPTS and j < max_j
                        and sess.run(pruning_obj._sparsity) > PRUNING_CKPTS[j]):
                    tf.logging.info(
                        'Saving pruning ckpt: ' + str(PRUNING_CKPTS[j]))
                    new_dir = pruning_dir + str(PRUNING_CKPTS[j]) + '/'
                    utils.duplicate_saved(MODEL_DIR, new_dir)
                    j += 1


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
    pruning_hparams = pruning.get_pruning_hparams().parse(
        PRUNING_HPARAMS)
    pruning_obj = pruning.Pruning(
        pruning_hparams, global_step=global_step)
    mask_update_op = pruning_obj.conditional_mask_update_op()
    pruning_obj.add_pruning_summaries()
    return train_op, mask_update_op, pruning_obj, loss


def calc_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
#    lr = 0.02
    lr = utils.manual_stepping(global_step, boundaries=[10000, 20000],
        rates = [0.2, 0.1, 0.05])

    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999,
        global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    MODEL_DIR = HOME_DIR + MODEL_NAME
    tf.logging.info(MODEL_DIR)
    train_mnist()
