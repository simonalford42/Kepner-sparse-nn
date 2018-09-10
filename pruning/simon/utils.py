from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import tensorflow as tf
import shutil, errno
sys.path.append(os.path.abspath('..'))

def manual_stepping(global_step, boundaries, rates, warmup=False):
    """ Manually stepped learning rate schedule.  This function provides fine
    grained control over learning rates.  One must specify a sequence of
    learning rates as well as a set of integer steps at which the current
    learning rate must transition to the next.  For example, if boundaries =
    [5, 10] and rates = [.1, .01, .001], then the learning rate returned by
    this function is .1 for global_step=0,...,4, .01 for global_step=5...9, and
    .001 for global_step=10 and onward.

    Args:

    global_step: int64 (scalar) tensor representing global step.

    boundaries: a list of global steps at which to switch learning rates.  This
    list is assumed to consist of increasing positive integers.

    rates: a list of (float) learning rates corresponding to intervals between
    the boundaries.  The length of this list must be exactly len(boundaries) +
    1.

    warmup: Whether to linearly interpolate learning rate for steps in [0,
    boundaries[0]].  Returns: a (scalar) float tensor representing learning
    rate

    Raises: ValueError: if one of the following checks fails: 1. boundaries is
    a strictly increasing list of positive integers 2. len(rates) ==
    len(boundaries) + 1 3.  boundaries[0] != 0 """

    if any([b < 0 for b in boundaries]) or any(
            [not isinstance(b, int) for b in boundaries]):
        raise ValueError('boundaries must be a list of positive integers')

    if any([bnext <= b for bnext, b in zip(boundaries[1:], boundaries[:-1])]):
        raise ValueError('Entries in boundaries must be strictly increasing.')

    if any([not isinstance(r, float) for r in rates]):
        raise ValueError('Learning rates must be floats')

    if len(rates) != len(boundaries) + 1:
        raise ValueError('Number of provided learning rates must exceed '
                         + 'number of boundary points by exactly 1.')

    if boundaries and boundaries[0] == 0:
        raise ValueError('First step cannot be zero.')

    if warmup and boundaries:
        slope = (rates[1] - rates[0]) * 1.0 / boundaries[0]
        warmup_steps = range(boundaries[0])
        warmup_rates = [rates[0] + slope * step for step in warmup_steps]
        boundaries = warmup_steps + boundaries
        rates = warmup_rates + rates[1:]
    else:
        boundaries = [0] + boundaries
    num_boundaries = len(boundaries)
    rate_index = tf.reduce_max(tf.where(tf.greater_equal(
        global_step, boundaries), list(range(num_boundaries)),
        [0] * num_boundaries))

    return tf.reduce_sum(rates * tf.one_hot(rate_index, depth=num_boundaries),
        name='learning_rate')


def copy(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


def update_ckpt_file(path):
    """
    This is used to update checkpoint file paths of moved directories so that
    you can restore from the checkpoints after moving the directory.
    """
    with open(path + '/checkpoint', 'r') as old:
        with open(path + '/checkpoint2', 'w+') as new:
            lines = old.readlines()
            for line in lines:
                a = line.index('\"')
                b = line.index('model.ckpt')
                new_line = line[:a+1] + path + line[b:]
                new.write(new_line)
#                print(new_line)

    os.remove(path + '/checkpoint')
    os.rename(path + '/checkpoint2', path + '/checkpoint')

def duplicate_saved(path, new_path):
    copy(path, new_path)
    update_ckpt_file(new_path)

if __name__ == '__main__':
    p = '/home/gridsan/salford/Kepner-sparse-nn/saved/'
    duplicate_saved(p + 'prune3/cifar/', p + '/cifar10_bang/cifar10/')
