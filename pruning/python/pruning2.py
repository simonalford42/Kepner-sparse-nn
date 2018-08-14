# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("/home/gridsan/salford/tf/model_pruning/")
from python import pruning_utils
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


def apply_mask(x, scope=''):
  """Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope. Defaults to "".
  Returns:
    Tensor representing masked_weights
  """

  mask = pruning_utils.weight_mask_variable(x, scope)
  threshold = pruning_utils.weight_threshold_variable(x, scope)
  # Add masked_weights in the weights namescope so as to make it easier
  # for the quantization library to add quant ops.
  masked_weights = math_ops.multiply(mask, x, _MASKED_WEIGHT_NAME)

  # Make sure the mask for a given variable are not added multiple times to the
  # collection. This is particularly important when applying mask to RNN's
  # weight variables
  if mask not in ops.get_collection_ref(_MASK_COLLECTION):
    ops.add_to_collection(_THRESHOLD_COLLECTION, threshold)
    ops.add_to_collection(_MASK_COLLECTION, mask)
    ops.add_to_collection(_MASKED_WEIGHT_COLLECTION, masked_weights)
    ops.add_to_collection(_WEIGHT_COLLECTION, x)
  return masked_weights


def get_masked_weights():
  return ops.get_collection(_MASKED_WEIGHT_COLLECTION)


def get_masks():
  return ops.get_collection(_MASK_COLLECTION)


def get_thresholds():
  return ops.get_collection(_THRESHOLD_COLLECTION)


def get_weights():
  return ops.get_collection(_WEIGHT_COLLECTION)


def get_weight_sparsity():
  """Get sparsity of the weights.

  Args:
    None

  Returns:
    A list containing the sparsity of each of the weight tensors
  """
  masks = get_masks()
  return [nn_impl.zero_fraction(mask) for mask in masks]


def get_pruning_hparams():
  """
 Returns:
    tf.HParams object initialized to default values

  """
  return hparam.HParams(
      name='model_pruning',
      begin_pruning_step=100,
      end_pruning_step=-400,
      do_not_prune=[''],
      pruning_frequency=20,
      nbins=256,
      threshold=0.1
      )

class Pruning(object):

  def __init__(self, spec=None, global_step=None, sparsity=None):
    """Set up the specification for model pruning.

    If a spec is provided, the sparsity is set up based on the sparsity_function
    in the spec. The effect of sparsity_function is overridden if the sparsity
    variable is passed to the constructor. This enables setting up arbitrary
    sparsity profiles externally and passing it to this pruning functions.

    Args:
      spec: Pruning spec as defined in pruning.proto
      global_step: A tensorflow variable that is used while setting up the
        sparsity function
      sparsity: A tensorflow scalar variable storing the sparsity
    """
    # Pruning specification
    self._spec = spec if spec else get_pruning_hparams()

    # A tensorflow variable that tracks the sparsity function.
    # If not provided as input, the graph must already contain the global_step
    # variable before calling this constructor.
    self._global_step = self._setup_global_step(global_step)

    # List of tensorflow assignments ops for new masks and thresholds
    self._assign_ops = []

    # Tensorflow variable keeping track of the last global step when the masks
    # were updated
    self._last_update_step = self._setup_last_update_step()

  def _setup_global_step(self, global_step):
    graph_global_step = global_step
    if graph_global_step is None:
      graph_global_step = training_util.get_global_step()

    return math_ops.cast(graph_global_step, dtypes.int32)

  def _setup_last_update_step(self):
    # use_resource=self._spec.use_tpu replaced with False
    with variable_scope.variable_scope(
        self._spec.name, use_resource=False) as scope:
      try:
        last_update_step = variable_scope.get_variable(
            'last_mask_update_step', [],
            initializer=init_ops.zeros_initializer(),
            trainable=False,
            dtype=dtypes.int32)
      except ValueError:
        scope.reuse_variables()
        last_update_step = variable_scope.get_variable(
            'last_mask_update_step', dtype=dtypes.int32)
    return last_update_step

  def _exists_in_do_not_prune_list(self, tensor_name):
    do_not_prune_list = self._spec.do_not_prune
    if not do_not_prune_list[0]:
      return False
    for layer_name in do_not_prune_list:
      if tensor_name.find(layer_name) != -1:
        return True

    return False

  def _update_mask(self, weights, threshold):
    """Updates the mask for a given weight tensor.

    Given the threshold, returns new mask which has 0 or 1 where weights are pruned or kept.

    Args:
      weights: The weight tensor that needs to be masked.
      threshold: The current threshold value.

    Returns:
      new_mask: A numpy array of the same size and shape as weights containing
        0 or 1 to indicate which of the values in weights falls below
        the threshold
    """
    print('start update mask')
    with ops.name_scope(weights.op.name + '_pruning_ops'):
      abs_weights = math_ops.abs(weights)
      new_mask = math_ops.cast(
          math_ops.greater(abs_weights, threshold), dtypes.float32)
    print('end update mask')
    return threshold, new_mask

  def _get_mask_assign_ops(self):
    # Make sure the assignment ops have not already been added to the list
    if self._assign_ops:
      raise ValueError(
          'Assign op list not empty. _get_mask_assign_ops() called twice?')

    masks = get_masks()
    weights = get_weights()
    thresholds = get_thresholds()

    if len(masks) != len(thresholds):
      raise ValueError(
          'Number of masks %s and number of thresholds %s mismatch' %
          (len(masks), len(thresholds)))

    for index, mask in enumerate(masks):
      threshold = thresholds[index]
      weight = weights[index]
      is_partitioned = isinstance(weight, variables.PartitionedVariable)
      if is_partitioned:
        weight = weight.as_tensor()

      if self._spec.do_not_prune:
        if self._exists_in_do_not_prune_list(mask.name):
          continue

      new_threshold, new_mask = self._update_mask(weight, threshold)
      self._assign_ops.append(
          pruning_utils.variable_assign(threshold, new_threshold))

      self._assign_ops.append(
          pruning_utils.partitioned_variable_assign(mask, new_mask)
          if is_partitioned else pruning_utils.variable_assign(mask, new_mask))

  def mask_update_op(self):
    with ops.name_scope(self._spec.name):
      if not self._assign_ops:
        self._get_mask_assign_ops()
      with ops.control_dependencies([
          state_ops.assign(
              self._last_update_step,
              self._global_step,
              name='last_mask_update_step_assign')
      ]):
        with ops.control_dependencies(self._assign_ops):
          logging.info('Updating masks.')
          return control_flow_ops.no_op('mask_update')

  def conditional_mask_update_op(self):

    def maybe_update_masks():
      with ops.name_scope(self._spec.name):
        is_step_within_pruning_range = math_ops.logical_and(
            math_ops.greater_equal(self._global_step,
                                   self._spec.begin_pruning_step),
            # If end_pruning_step is negative, keep pruning forever!
            math_ops.logical_or(
                math_ops.less_equal(self._global_step,
                                    self._spec.end_pruning_step),
                math_ops.less(self._spec.end_pruning_step, 0)))
        is_pruning_step = math_ops.less_equal(
            math_ops.add(self._last_update_step, self._spec.pruning_frequency),
            self._global_step)
        return math_ops.logical_and(is_step_within_pruning_range,
                                    is_pruning_step)

    def mask_update_op():
      return self.mask_update_op()

    def no_update_op():
      return control_flow_ops.no_op()

    return control_flow_ops.cond(maybe_update_masks(), mask_update_op,
                                 no_update_op)

  def add_pruning_summaries(self):
    """Adds summaries for this pruning spec.

    Args: none

    Returns: none
    """
    with ops.name_scope(self._spec.name + '_summaries'):
      summary.scalar('last_mask_update_step', self._last_update_step)
      masks = get_masks()
      thresholds = get_thresholds()
      for index, mask in enumerate(masks):
        if not self._exists_in_do_not_prune_list(mask.name):
          summary.scalar(mask.name + '/sparsity', nn_impl.zero_fraction(mask))
          summary.scalar(thresholds[index].op.name + '/threshold',
                         thresholds[index])

  def print_hparams(self):
    logging.info(self._spec.to_json())
