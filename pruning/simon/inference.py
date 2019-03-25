import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from python import pruning


def fc_layer(last_layer, shape, scope):
    weights = _variable_with_weight_decay(
        'weights', shape=shape, stddev=0.04, wd=0.004)
    biases = _variable_on_cpu(
        'biases', shape[1], tf.constant_initializer(0.1))
    pruned_weights = pruning.apply_mask(weights, scope)
    _mask_summary(pruned_weights)
    dense = tf.nn.relu(
        tf.matmul(last_layer, pruned_weights) + biases,
        name=scope.name)
    _activation_summary(dense)
    return dense


def conv_layer(last_layer, shape, scope):
    kernel = _variable_with_weight_decay(
            'weights', shape=shape, stddev=5e-2, wd=0.0)
    pruned_kernel = pruning.apply_mask(kernel, scope)
    _mask_summary(pruned_kernel)
    conv = tf.nn.conv2d(last_layer, pruned_kernel, [1, 1, 1, 1],
                        padding='SAME')
    biases = _variable_on_cpu('biases', [shape[3]],
                              tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv)
    return conv

def lenet3000_1000_500_100(features):
    with tf.variable_scope('dense1') as scope:
        layer = fc_layer(features, [784, 1000], scope)

    with tf.variable_scope('dense2') as scope:
        layer2 = fc_layer(layer, [1000, 5000], scope)

    with tf.variable_scope('dense3') as scope:
        layer3 = fc_layer(layer2, [500, 100], scope)

    with tf.variable_scope('logits') as scope:
        logits = fc_layer(layer3, [100, 10], scope)

    return logits


def get_lenet300_100_mime(ratio):

    def lenet300_100_mime(features):
        input_layer = features

        with tf.variable_scope('dense1') as scope:
            layer = fc_layer(features, [784, int(300*ratio)], scope)

        with tf.variable_scope('dense2') as scope:
            layer2 = fc_layer(layer, [int(300*ratio), 100], scope)

        with tf.variable_scope('logits') as scope:
            logits = fc_layer(layer2, [100, 10], scope)

        return logits

    return lenet300_100_mime


def get_lenet5_mime(ratio):

    def lenet5_mime(features):
        l1_filters = 20
        l2_filters = 50*ratio
        input_layer = tf.reshape(features, [-1, 28, 28, 1])

        with tf.variable_scope('conv1') as scope:
            conv1 = conv_layer(input_layer, [5, 5, 1, l1_filters], scope)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(
                pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        with tf.variable_scope('conv2') as scope:
            conv2 = conv_layer(norm1, [5, 5, l1_filters, l2_filters], scope)

        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        # 2450 * ratio
        pool2_flat_size = np.prod(pool2.shape.as_list()[1:])

        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [-1, pool2_flat_size])

        # fully connected layer 1
        with tf.variable_scope('dense1') as scope:
            layer = fc_layer(reshape, [pool2_flat_size, 384], scope)

        # fully connected layer 2
        with tf.variable_scope('logits') as scope:
            logits = fc_layer(layer, [384, 10], scope)

        return logits

    return lenet5_mime


def get_lenet5_cifar_mime(ratio):
    def lenet5_cifar(images):
        """
        images: images returned from distorted_inputs() or inputs().
        """
        l1_kernels = 20
        l2_kernels = 50*ratio

        with tf.variable_scope('conv1') as scope:
            conv1 = conv_layer(images, [5, 5, 3, l1_kernels], scope)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        with tf.variable_scope('conv2') as scope:
            conv2 = conv_layer(norm1, [5, 5, l1_kernels, l2_kernels], scope)

        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        # 1800 * ratio
        pool2_flat_size = np.prod(pool2.shape.as_list()[1:])

        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [-1, pool2_flat_size])

        with tf.variable_scope('dense1') as scope:
            dense1 = fc_layer(reshape, [pool2_flat_size, 384], scope)

        with tf.variable_scope('logits') as scope:
            logits = fc_layer(dense1, [384, 10], scope)

        return logits

    return lenet5_cifar

def get_lenet5_cifar_mime2(ratio):
    def lenet5_cifar(images):
        """
        images: images returned from distorted_inputs() or inputs().
        """
        l1_kernels = 20
        l2_kernels = 50*ratio

        with tf.variable_scope('conv1') as scope:
            conv1 = conv_layer(images, [5, 5, 3, l1_kernels], scope)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        with tf.variable_scope('conv2') as scope:
            conv2 = conv_layer(norm1, [5, 5, l1_kernels, l2_kernels], scope)

        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        # 1800 * ratio
        pool2_flat_size = np.prod(pool2.shape.as_list()[1:])

        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [-1, pool2_flat_size])

        with tf.variable_scope('fc1') as scope:
            dense1 = fc_layer(reshape, [pool2_flat_size, 384], scope)

        with tf.variable_scope('logits') as scope:
            logits = fc_layer(dense1, [384, 10], scope)

        return logits

    return lenet5_cifar


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


lenet300_100 = get_lenet300_100_mime(1)
lenet5 = get_lenet5_mime(1)
lenet5_cifar = get_lenet5_cifar_mime(1)
