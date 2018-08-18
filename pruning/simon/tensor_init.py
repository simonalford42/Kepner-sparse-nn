from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
print(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../..'))
from x_net import x_net

def rand_small_31():
    with tf.get_default_graph().as_default():
        arr = np.array([1,0], dtype=np.float32)
        dense1_mask = tf.convert_to_tensor(
            np.random.choice(
                arr, size=(784,300), p = [1./10, 9./10]))
        dense2_mask = tf.convert_to_tensor(
            np.random.choice(
                arr, size=(300,100), p = [1./10, 9./10]))
        init_dict = {'dense1': {'mask': dense1_mask}, 'dense2':
            {'mask': dense2_mask}}
    return init_dict

def rand_big_31():
    with tf.get_default_graph().as_default():
        arr = np.array([1,0], dtype=np.float32)
        dense1_mask = tf.convert_to_tensor(
            np.random.choice(
                arr, size=(784,3000), p = [1./10, 9./10]))
        dense2_mask = tf.convert_to_tensor(
            np.random.choice(
                arr, size=(3000,100), p = [1./10, 9./10]))
        init_dict = {'dense1': {'mask': dense1_mask}, 'dense2':
            {'mask': dense2_mask}}
    return init_dict

def emr_big_31():
    """
    To mimic lenet 300-100 with 784 input nodes, creates lenet 300-100 with
    800 input nodes using the k/emr network for the first two layers and a
    fully connected last layer and then 300, 100 in between.

    In this one, we have the same number connections as lenet300_100.
    """
    N = [[10, 10]]
    B = [8, 30, 1]
    layers, info = x_net.kemr_net(N, B)
    layers[0] = layers[0][16:]

    for layer in layers:
        print(layer.shape)

    print(info)

    init_dict = {'dense1': {'mask': layers[0]}, 'dense2': {'mask': layers[1]}}

    mask_summaries(N, B, info)

    return init_dict


def emr_small_31():
    """
    To mimic lenet 300-100 with 784 input nodes, creates lenet 300-100 with
    800 input nodes using the k/emr network for the first two layers and a
    fully connected last layer and then 300, 100 in between.

    This one is equivalent to a 90% sparse version of lenet300_100 (has same 
    total connections possible).
    """
    N = [[10, 10]]
    B = [8, 3, 1]
    layers, info = x_net.kemr_net(N, B)
    layers[0] = layers[0][16:]

    for layer in layers:
        print(layer.shape)

    print(info)

    init_dict = {'dense1': {'mask': layers[0]}, 'dense2': {'mask': layers[1]}}

    mask_summaries(N, B, info)

    return init_dict

def mask_summaries(N=None, B=None, info=None):
    with tf.name_scope('kemr_summaries'):
        if N:
            tf.summary.text('radix_lists', tf.convert_to_tensor(str(N)))
        if B:
            tf.summary.text('B list', tf.convert_to_tensor(str(B)))
        if info:
            tf.summary.text('shape', tf.convert_to_tensor(str(info['shape'])))
            tf.summary.text('paths', tf.convert_to_tensor(str(info['paths'])))
            tf.summary.text('sparsities',
                tf.convert_to_tensor(str(info['sp'])))
            tf.summary.text('avg sparsity',
                tf.convert_to_tensor(str(info['avg_sp'])))
            tf.summary.text('connections',
                tf.convert_to_tensor(str(info['conns'])))
            tf.summary.text('connections per neuron',
                tf.convert_to_tensor(str(info['cpn'])))
            tf.summary.text('avg_cpn',
                tf.convert_to_tensor(str(info['avg_cpn'])))

if __name__ == '__main__':
   layers = emr_big_31()


