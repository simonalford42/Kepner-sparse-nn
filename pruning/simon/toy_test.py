import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
import inference as infer
from python import pruning
from python.layers import core_layers as core

from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util

_MASK_COLLECTION = core.MASK_COLLECTION
_THRESHOLD_COLLECTION = core.THRESHOLD_COLLECTION
_MASKED_WEIGHT_COLLECTION = core.MASKED_WEIGHT_COLLECTION
_WEIGHT_COLLECTION = core.WEIGHT_COLLECTION
_MASKED_WEIGHT_NAME = core.MASKED_WEIGHT_NAME

def inference(a):
    with tf.variable_scope('logits') as scope:
#       b = tf.get_variable(
#           'b',shape=(10,1), initializer=tf.truncated_normal_initializer(
#               stddev=0.1, dtype=tf.float32))
        b = infer._variable_with_weight_decay('b',(10,1),0.1, 0.0)
        p = apply_mask(b, scope)
    return b


def weight_threshold_variable(var, scope):
  with variable_scope.variable_scope(scope):
    threshold = tf.get_variable(
        'threshold', [],
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        dtype=var.dtype)
    return threshold


def weight_mask_variable(var, scope):
  with variable_scope.variable_scope(scope):
    mask = tf.get_variable(
        'mask',
        var.get_shape(),
        initializer=init_ops.ones_initializer(),
        trainable=False,
        dtype=var.dtype)
  return mask


def apply_mask(x, scope=''):

    mask = weight_mask_variable(x, scope)
    threshold = weight_threshold_variable(x, scope)
    masked_weights = math_ops.multiply(mask, x, _MASKED_WEIGHT_NAME)

    if mask not in ops.get_collection_ref(_MASK_COLLECTION):
        ops.add_to_collection(_THRESHOLD_COLLECTION, threshold)
        ops.add_to_collection(_MASK_COLLECTION, mask)
        ops.add_to_collection(_MASKED_WEIGHT_COLLECTION, masked_weights)
        ops.add_to_collection(_WEIGHT_COLLECTION, x)

    return masked_weights

def train():
    with tf.Graph().as_default():
        inp = tf.Variable(5)
        outp = inference(inp)
        masks = tf.get_collection(core.MASK_COLLECTION)
        print('masks: ' + str(masks))
        INIT_DICT = {'logits':{'b': tf.convert_to_tensor(np.ones((10,1),dtype=np.float32)), 'mask': tf.convert_to_tensor(np.ones((10,1),dtype=np.float32))}}
        assign_ops = []
        for scope in INIT_DICT:
            print('scope= ' + scope)
            var_dict = INIT_DICT[scope]
            with tf.variable_scope(scope, reuse=True):
                for var in INIT_DICT[scope]:
                    print('var = ' + var)
                    assign_op = tf.assign(tf.get_variable(var), var_dict[var])
                    assign_ops.append(assign_op)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if INIT_DICT is not None:
                sess.run(assign_ops)
                for scope in INIT_DICT.keys():
                    with tf.variable_scope(scope, reuse=True):
                        for var in INIT_DICT[scope].keys():
                            print('assigned ' + scope + '/' + var + ': '
                                  + str(sess.run(tf.get_variable(var))))
            print(sess.run(outp))


if __name__ == '__main__':
    train()
