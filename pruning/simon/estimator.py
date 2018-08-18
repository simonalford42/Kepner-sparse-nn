from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from python import pruning as pruning
BATCH_SIZE = 100

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

def get_lenet5_model_fn(pruning_hparam_args):

    def lenet5_model_fn(features, labels, mode):
        l1_filters = 20
        l2_filters = 50
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                    'weights', shape=[5, 5, 1, l1_filters], stddev=5e-2, wd=0.0)

            conv = tf.nn.conv2d(
                    input_layer,
                    pruning.apply_mask(kernel, scope),
                    [1, 1, 1, 1],
                    padding='SAME')
            biases = _variable_on_cpu('biases', [l1_filters], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(
                conv1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')

        # norm1
        norm1 = tf.nn.lrn(
                pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(
                    'weights', shape=[5, 5, l1_filters, l2_filters], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                    norm1, pruning.apply_mask(kernel, scope), [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [l2_filters], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)

        # norm2
        norm2 = tf.nn.lrn(
                conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(
                norm2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

        pool2_flat_size = np.prod(pool2.shape.as_list()[1:])

        # fully connected layer 1
        with tf.variable_scope('fc1') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [-1, pool2_flat_size])
            weights = _variable_with_weight_decay(
                    'weights', shape=[pool2_flat_size, 384], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(
                    tf.matmul(reshape, pruning.apply_mask(weights, scope)) + biases,
                    name=scope.name)
            _activation_summary(fc1)

        # fully connected layer 2
        with tf.variable_scope('logits') as scope:
            weights = _variable_with_weight_decay(
                    'weights', [384, 10], stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu('biases', [10],
                                                                tf.constant_initializer(0.0))
            logits = tf.add(
                    tf.matmul(fc1, pruning.apply_mask(weights, scope)),
                    biases,
                    name=scope.name)
            _activation_summary(logits)

        return estimator_fn_complete(pruning_hparam_args, features,
                labels, mode, logits)

    return lenet5_model_fn

def get_lenet5_model_fn2(pruning_hparam_args):

    def lenet5_model_fn(features, labels, mode):
        l1_filters = 20
        l2_filters = 50
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                    'weights', shape=[5, 5, 1, l1_filters], stddev=5e-2, wd=0.0)
            pruned_kernel = pruning.apply_mask(kernel, scope)
            _mask_summary(pruned_kernel)
            conv = tf.nn.conv2d(
                    input_layer,
                    pruned_kernel,
                    [1, 1, 1, 1],
                    padding='SAME')
            biases = _variable_on_cpu('biases', [l1_filters], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)

        # pool1
        pool1 = tf.nn.max_pool(
                conv1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')

        # norm1
        norm1 = tf.nn.lrn(
                pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(
                    'weights', shape=[5, 5, l1_filters, l2_filters], stddev=5e-2, wd=0.0)
            pruned_kernel = pruning.apply_mask(kernel, scope)
            _mask_summary(pruned_kernel)
            conv = tf.nn.conv2d(
                    norm1,
                    pruned_kernel,
                    [1, 1, 1, 1],
                    padding='SAME')
            biases = _variable_on_cpu('biases', [l2_filters], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)

        # norm2
        norm2 = tf.nn.lrn(
                conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(
                norm2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

        pool2_flat_size = np.prod(pool2.shape.as_list()[1:])
        print(pool2_flat_size)

        # fully connected layer 1
        with tf.variable_scope('fc1') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [-1, pool2_flat_size])
            weights = _variable_with_weight_decay(
                    'weights', shape=[pool2_flat_size, 384], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            fc1 = tf.nn.relu(
                    tf.matmul(reshape, pruned_weights) + biases,
                    name=scope.name)
            _activation_summary(fc1)

        # fully connected layer 2
        with tf.variable_scope('logits') as scope:
            weights = _variable_with_weight_decay(
                    'weights', [384, 10], stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu('biases', [10],
                                                                tf.constant_initializer(0.0))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            logits = tf.add(
                    tf.matmul(fc1, pruned_weights),
                    biases,
                    name=scope.name)
            _activation_summary(logits)

        return estimator_fn_complete(pruning_hparam_args, features,
                labels, mode, logits)

    return lenet5_model_fn


def estimator_fn_complete(pruning_hparam_args, features, labels,
        mode, logits):
    '''
    this method does the rest of the work needd to define the model_fn
    for an estimator. It is here because multiple models share the same ending details.
    '''
    predictions = {
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    pruning_hparams = pruning.get_pruning_hparams() \
            .parse(pruning_hparam_args)
    global_step = tf.train.get_global_step()
    pruning_obj = pruning.Pruning(
            pruning_hparams, global_step=global_step)
    pruning_obj.add_pruning_summaries()

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

    loss = calc_loss(logits, labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        acc = tf.reduce_mean(tf.cast(tf.equal(predictions['classes'], tf.cast(labels, tf.int64)), tf.float32))
        tf.summary.scalar('accuracy', acc)

        train_op = train(loss, global_step)
        with tf.control_dependencies([train_op]):
            mask_update_op = pruning_obj.conditional_mask_update_op()

        # mask_update_op depends on train_op, so this should call train_op
        # and mask_update op in succession while training
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=mask_update_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_lenet300_100_test(pruning_hparam_args):
    def lenet300_100_test(features, labels, mode):
        input_layer = features['x']

        dense1 = tf.layers.dense(
                inputs=input_layer,
                units=300,
                activation=tf.nn.relu)
        _activation_summary(dense1)

        dense2 = tf.layers.dense(
                inputs=dense1,
                units=100,
                activation=tf.nn.relu)
        _activation_summary(dense2)

        logits = tf.layers.dense(
                inputs=dense2,
                units=10)
        _activation_summary(logits)

        return estimator_fn_complete(pruning_hparam_args, features, labels, mode, logits)

    return lenet300_100_test

def get_lenet300_100_model_fn(pruning_hparam_args):

    def lenet300_100_model_fn(features, labels, mode):
        input_layer = features['x']

        with tf.variable_scope('dense1') as scope:
            weights = _variable_with_weight_decay(
                    'weights', shape=[784, 300], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases',[300], tf.constant_initializer(0.1))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            dense1 = tf.nn.relu(
                        tf.matmul(input_layer, pruned_weights) + biases,
                        name=scope.name)
            _activation_summary(dense1)

        with tf.variable_scope('dense2') as scope:
            weights = _variable_with_weight_decay(
                    'weights', shape=[300, 100], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases',[100], tf.constant_initializer(0.1))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            dense2 = tf.nn.relu(
                        tf.matmul(dense1, pruned_weights) + biases,
                        name=scope.name)
            _activation_summary(dense2)

        with tf.variable_scope('logits') as scope:
            weights = _variable_with_weight_decay(
                    'weights', shape=[100, 10], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases',[10], tf.constant_initializer(0.1))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            logits = tf.nn.relu(
                        tf.matmul(dense2, pruned_weights) + biases,
                        name=scope.name)
            _activation_summary(logits)

        return estimator_fn_complete(pruning_hparam_args, features, labels, mode, logits)

    return lenet300_100_model_fn

def get_baby_net_model_fn(pruning_hparam_args):
    def baby_net_model_fn(features, labels, mode):
        layer1_size = 50
        # Input Layer
        input_layer = features['x']
        input_size = np.prod(input_layer.shape.as_list()[1:])

        # fully connected layer 1
        with tf.variable_scope('fc1') as scope:
            weights = _variable_with_weight_decay(
                    'weights', shape=[input_size, layer1_size],
                    stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [layer1_size],
                    tf.constant_initializer(0.1))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            fc1 = tf.nn.relu(
                    tf.matmul(input_layer, pruned_weights) + biases,
                    name=scope.name)
            _activation_summary(fc1)

        # logits layer
        with tf.variable_scope('logits') as scope:
            weights = _variable_with_weight_decay(
                    'weights', [layer1_size, 10], stddev=1 / 192.0, wd=0.0)
            biases = _variable_on_cpu('biases', [10],
                                                                tf.constant_initializer(0.0))
            pruned_weights = pruning.apply_mask(weights, scope)
            _mask_summary(pruned_weights)
            logits = tf.add(
                    tf.matmul(fc1, pruned_weights),
                    biases,
                    name=scope.name)
            _activation_summary(logits)

        return estimator_fn_complete(pruning_hparam_args, features,
                labels, mode, logits)

    return baby_net_model_fn

def calc_loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                        of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.
    Returns:
        train_op: op for training.
    """
    lr = 0.1
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
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                                                                                global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def _mask_summary(x):
    tensor_name = x.op.name
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(tensor_name + '/mask', x)

def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/activation sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(name, shape,
                                                 tf.truncated_normal_initializer(
                                                         stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

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
