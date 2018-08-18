import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from python import pruning

def lenet3000_100(features):
    input_layer = features

    with tf.variable_scope('dense1') as scope:
        weights = _variable_with_weight_decay(
                'weights', shape=[784, 3000], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases',[3000], tf.constant_initializer(0.1))
        pruned_weights = pruning.apply_mask(weights, scope)
        _mask_summary(pruned_weights)
        dense1 = tf.nn.relu(
                    tf.matmul(input_layer, pruned_weights) + biases,
                    name=scope.name)
        _activation_summary(dense1)

    with tf.variable_scope('dense2') as scope:
        weights = _variable_with_weight_decay(
                'weights', shape=[3000, 100], stddev=0.04, wd=0.004)
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

        return logits

def lenet300_100(features):
    input_layer = features

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

        return logits

def lenet5(features):
    l1_filters = 20
    l2_filters = 50
    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

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

    return logits


def toy_numpy(features):
    input_layer = features
    with tf.variable_scope('logits') as scope:
        W = _variable_with_weight_decay('W', [784, 10], stddev=1/192.0, wd=0.0)
        pruned_weights = pruning.apply_mask(W, scope)
        logits = tf.matmul(input_layer, W)
        _activation_summary(logits)
    return logits


def _mask_summary(x):
    tensor_name = x.op.name
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(tensor_name, x)

def _activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/activation sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay_init(name, shape, wd, init_tensor):
    """Helper to create an initialized Variable with weight decay. Adds
    Capability to initialize with some other variable.

    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        init_tensor: initialize variable to the value of this tensor.

    Returns:
        Variable Tensor
    """
    pass

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
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(
        stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
