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
sys.path.append("/home/gridsan/salford/tf/model_pruning/")
from python import pruning as pruning
from python.layers import core_layers as core
import inference
from mnist_input import MnistData
tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 100

HOME_DIR = '/home/gridsan/salford/tf/saved/'
# MODEL_NAME = 'lenet5_rt'
# MODEL_NAME = 'lenet300_100_rt'
MODEL_NAME = 'toy_numpy'
MODEL_DIR = HOME_DIR + MODEL_NAME
# MODEL_FN = inference.lenet5_inference
# MODEL_FN = inference.lenet300_100_inference
MODEL_FN = inference.toy_numpy_inference

TRAIN_STEPS = 500
assert TRAIN_STEPS % 100 == 0
LEARNING_RATE = 0.1
FLAGS = None

TEST_SET_SIZE = 10000
TEST_FREQ = 200
assert TEST_FREQ % 100 == 0

RESTORE_MODEL = 'lenet300_100_prune2'
RESTORE_DIR = HOME_DIR + RESTORE_MODEL

# pruning params
PRUNING_ON = False
BEGIN_PRUNING_STEP = 2000
END_PRUNING_STEP = 5000
INITIAL_SPARSITY = 0.0
TARGET_SPARSITY = 0.9
SPARSITY_FUNCTION_BEGIN_STEP = BEGIN_PRUNING_STEP
SPARSITY_FUNCTION_END_STEP = END_PRUNING_STEP
PRUNING_FREQUENCY = 200
assert PRUNING_FREQUENCY % 100 == 0

# restore dict for numpy arrays
# structure: scope: dict with structure var_name: value

def train_mnist(num_steps=TRAIN_STEPS):
    with tf.Graph().as_default():
        INIT_DICT = {'logits': {'mask:0': tf.convert_to_tensor(np.ones((784, 10)))}}
        input_pipe = MnistData(BATCH_SIZE)
        train_features, train_labels = input_pipe.build_train_data_tensor()

        train_op, mask_update_op, loss = train_graph(train_features, train_labels)

        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
        new_masks = tf.get_collection(core.MASK_COLLECTION)
        print('is inited: ' + str(tf.is_variable_initialized(new_masks[0])))
        print('new_masks: ' + str(new_masks))
        assign_ops = []
        for scope in INIT_DICT:
            print('scope= ' + scope)
            var_dict = INIT_DICT[scope]
            with tf.variable_scope(scope, reuse=True):
                for var in INIT_DICT[scope]:
                    print('var = ' + var)
                    assign_op = tf.assign(tf.get_variable(var), var_dict[var])
                    assign_ops.append(assign_op)

        if checkpoint is None:
            print('No checkpoint found; training from scratch')
            global_step = 0
        else:
            saver = tf.train.Saver()
            # Assuming model_checkpoint_path looks something like:
            #       /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = float(checkpoint.model_checkpoint_path
                                .split('/')[-1].split('-')[-1])

        checkpoint_saver = tf.train.Saver()
        checkpoint_hook = basic_session_run_hooks.CheckpointSaverHook(
            MODEL_DIR, save_steps = TEST_FREQ)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -0.5 + global_step

            def before_run(self, run_context):
                # this is a hack so that it correctly counts
                # (trainop, mask_update_op) as one step
                self._step += 0.5
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)    # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 100 == 0:
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
            print('made it 1')
            sess.run(tf.global_variables_initializer())
            print('made it 2')
            print('is inited: ' + str(sess.run(tf.is_variable_initialized(new_masks[0]))))

            if checkpoint is not None:
                saver.restore(sess, checkpoint.model_checkpoint_path)
            if INIT_DICT is not None:
                sess.run(assign_ops)
                for scope in INIT_DICT.keys():
                    with tf.variable_scope(scope, reuse=True):
                        for var in INIT_DICT[scope].keys():
                            print('assigned ' + scope + '/' + var + ': ' 
                                  + str(sess.run(tf.get_variable(var))))

            for i in range(TRAIN_STEPS):
                sess.run(train_op)
                sess.run(mask_update_op)


def train_graph(features, labels):
    global_step = tf.contrib.framework.get_or_create_global_step()

    labels = tf.cast(labels, tf.int64)

    logits = MODEL_FN(features)

    loss = calc_loss(logits, labels)

    predictions = tf.argmax(input=logits, axis=1)

    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(predictions, labels), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    train_op = train(loss, global_step)

    pruning_hparams = pruning.get_pruning_hparams().parse(
        FLAGS.pruning_hparams)
    pruning_obj = pruning.Pruning(
        pruning_hparams, global_step=global_step)
    mask_update_op = pruning_obj.conditional_mask_update_op()
    pruning_obj.add_pruning_summaries()

    return train_op, mask_update_op, loss


def calc_loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
    lr = LEARNING_RATE
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


def train_mnist_restore(num_steps=TRAIN_STEPS):
    with tf.Graph().as_default():
        input_pipe = MnistData(BATCH_SIZE)
        train_features, train_labels = input_pipe.build_train_data_tensor()

        train_op, mask_update_op, loss = train_graph(train_features, train_labels)

        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)

        if checkpoint is None:
            print('No checkpoint found; training from scratch')
            global_step = 0
        else:
            saver = tf.train.Saver()
            # Assuming model_checkpoint_path looks something like:
            #       /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = float(checkpoint.model_checkpoint_path
                                .split('/')[-1].split('-')[-1])

        new_masks = tf.get_collection(core.MASK_COLLECTION)
        print(str(new_masks))
#        saver2 = tf.train.Saver(var_list=new_masks)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -0.5 + global_step

            def before_run(self, run_context):
                # this is a hack so that it correctly counts
                # (trainop, mask_update_op) as one step
                self._step += 0.5
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)    # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 100 == 0:
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
                hooks=[_LoggerHook()],
                save_checkpoint_secs=60,
                config=tf.ConfigProto(log_device_placement=False)) as sess:
            if checkpoint is not None:
                saver.restore(sess, checkpoint.model_checkpoint_path)

            #saver2.restore(sess, tf.train.latest_checkpoint(RESTORE_DIR))
            print('initializing with variables from ' + RESTORE_DIR + ': ' +
                    str(new_masks))

            for i in range(TRAIN_STEPS):
                sess.run(train_op)
                sess.run(mask_update_op)


def main(arvg=None):
    train_mnist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pruning_hparams',
        type=str,
        default='name=' + MODEL_NAME + ','
        + 'begin_pruning_step=' + str(BEGIN_PRUNING_STEP) + ','
        + 'end_pruning_step=' + str(END_PRUNING_STEP)+ ','
        + 'initial_sparsity=' + str(INITIAL_SPARSITY)+ ','
        + 'target_sparsity=' + str(TARGET_SPARSITY) + ','
        + 'sparsity_function_begin_step=' + str(SPARSITY_FUNCTION_BEGIN_STEP)+ ','
        + 'sparsity_function_end_step=' + str(SPARSITY_FUNCTION_END_STEP)+ ','
        + 'pruning_frequency=' + str(PRUNING_FREQUENCY) + ','
        + 'pruning_on=' + str(PRUNING_ON),
        help="""Comma separated list of pruning-related hyperparameters"""  )

    FLAGS, unparsed = parser.parse_known_args()
    print('FLAGS: ' + str(FLAGS))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
