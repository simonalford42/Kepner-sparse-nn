from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
import argparse
sys.path.append("/home/gridsan/salford/tf/model_pruning/")
import estimator
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None

def lenet5(model_dir="lenet5_prune", ws = None):
    return tf.estimator.Estimator(
        model_fn=estimator.get_lenet5_model_fn2(FLAGS.pruning_hparams),
        model_dir="/home/gridsan/salford/tf/saved/" + model_dir + '/')


def lenet300_100(model_dir='lenet300_100', ws = None):
    return tf.estimator.Estimator(
        model_fn=estimator.get_lenet300_100_model_fn(FLAGS.pruning_hparams),
        model_dir="/home/gridsan/salford/tf/saved/" + model_dir + '/')


def baby_net(model_dir='baby_net_prune'):
    return tf.estimator.Estimator(
        model_fn=estimator.get_baby_net_model_fn(FLAGS.pruning_hparams),
        model_dir='/home/gridsan/salford/tf/saved/' + model_dir + '/')


def train_lenet(model, num_steps, test_freq=None):
    mnist = input_data.read_data_sets("/home/gridsan/salford/mnist/data/")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    steps = 0
    while steps < num_steps:
        to_go = min(num_steps - steps, test_freq)
        train(train_data, train_labels, model, to_go)
        test(eval_data, eval_labels, model)
        steps += to_go


def test_lenet(model):
    mnist = input_data.read_data_sets("/home/gridsan/salford/mnist/data/")
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    test(eval_data, eval_labels, model)


def train(train_data, train_labels, classifier, iters):
    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=estimator.BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=iters)


def test(eval_data, eval_labels, classifier):
    # test
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def main(argv=None):
#    train_lenet(lenet300_100('lenet300_100_mn2'), 5000, 300)
    train_lenet(lenet5('lenet5_mn2'), 5000, 300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pruning_hparams',
        type=str,
        default='name=default_model,begin_pruning_step=200000,'
                + 'end_pruning_step=500000,initial_sparsity=0.0,'
                + 'target_sparsity=0.0,sparsity_function_begin_step=200000,'
                + 'sparsity_function_end_step=500000',
        help="""Comma separated list of pruning-related hyperparameters"""  )

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
