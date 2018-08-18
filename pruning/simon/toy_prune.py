import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os
import argparse
sys.path.append(os.path.abspath('..'))
import tf as tf2
tf.logging.set_verbosity(tf.logging.INFO)

def lenet_weights(model_name):
    model_dir = '/home/gridsan/salford/tf/saved/' + model_name + '/'
    meta_file = 'model.ckpt-12001.meta'
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_dir + meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    layers = tf.trainable_variables()
    return layers

def do_stuff()
    layers = lenet_weights('lenet5_0.2')
    prune_params = {
            'sparsity_factors': {conv2d: 0.5, conv2d_1: 0.5},
            'acc_goal': 0.95,
            'sparsity_delta': 1}

def update(layers, prune_params):
    # returns (masks, updated_layers, sparsity
    # finds the right sparsity level which brings down the accuracy
    # of the model to the desired level of the original accuracy
    layer_thresholds = {}
    for layer in layers:
        name = layer.name
        factor = sparsity_factors[name] if name in sparsity_factors else 1
        sparsities=np.linspace(0, 100*sparsity_factors, 101)
        layer_thresholds[name] = thresholds(layer, sparsities)

    updated_layers, sparsity = binary_sparsity_search(layers, layer_thresholds, prune_params)

def binary_sparsity_search(layers, layer_thresholds, prune_params):
    min_sp = 0.0
    max_sp = 1.0
    last_sp = 0.0
    sp = 0.5
    sp_delta = prune_params['sparsity_delta']

    while abs(sp - last_sp) > sp_delta:
        updated_layers = [prune(layer, layer_thresholds[layer][sp])
                for layer in layers)
        last_sp = sp
        sp = (min_sp + max_sp) / 2.0


def thresholds(layer, sparsities):
    pass


def test(layers):
    # returns the accuracy of it
    pass

if __name__=='__main__':
    do_stuff()

