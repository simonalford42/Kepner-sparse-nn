from __future__ import division
import numpy as np
import fractions

def extended_mixed_radix_network(radix_lists):
  """
  Creates an X-net sparse network topology from the input parameters using Ryan's
  Extended Mixed-Radix method. See the documentation for a full 
  explanation of what is going on.

  Inputs:
  radix_lists - A list of lists of radices to be used for the [possibly extended]
      mixed radix system. The number of entries is the number of layers in the 
      network W that is output.

  Returns:
  layers - A list of numpy arrays where the i'th entry is the weight matrix W_i
      for the i'th layer.
  info   - A dict containing info about the layers with the following key/values:
      shape: a tuple containing the number of neurons for each layer. NOTE:
      this includes the input and output layers. as a result, 
      len(shape) = len(layers) + 1. That is, layers doesn't give the input weights.
      For the EMR method, the number of neurons is the same for each layer.
      paths: the number of possible paths between each input and output neuron.
      sparsity: the sparsity of the network (fraction of entries which are zero)
      connections_per_neuron: the average number of outgoing connections per neuron
      note that sparsity = 
      connections_per_neuron * num_neurons / total_connections_possible
  """
  num_layers = sum(len(radix_list) for radix_list in radix_lists) 
  # this is the number of neurons per layer.
  num_neurons = np.prod(radix_lists[0])

  # for all but last radix list, product of radices must equal num_neurons
  if not np.all(
      [num_neurons == np.prod(radix_list) for radix_list in radix_lists[:-1]]):
    raise ValueError('Product of radices for each radix list must equal'
        + 'number of neurons from first radix list for all but last radix list')
  # for last radix list, product of radices must divide num_neurons 
  if num_neurons % np.prod(radix_lists[-1]) != 0:
    raise ValueError('Product of radices for last radix list must divide'
        + 'number of neurons from first radix list')
  
  # the actual math part
  # The N x N identity matrix, used for constructing permutation matrices
  I = np.identity(num_neurons) 
  
  # calculate info 
  shape = tuple([num_neurons] * (num_layers + 1))
  paths  = np.prod([np.prod(radix_list) for radix_list in radix_lists[1:]])
  flattened_radices = [radix for radix_list in radix_lists for radix in radix_list]
  connections_per_neuron = np.average(flattened_radices) 
  total_neurons = np.sum(shape[:-1])
  total_connections_possible = num_neurons * num_neurons * num_layers
  sparsity_one = 1 - connections_per_neuron * total_neurons / total_connections_possible

  layers = [] # the output list containing W_i's
  # make layers
  for radix_list in radix_lists: 
    place_value = 1 
    for radix in radix_list:
      layer = np.sum(
          [np.roll(I, j * place_value, axis=1) for j in range(radix)], axis=0)
      layers.append(layer)
      place_value *= radix 
  
  # sparsity calculated manually (much slower, should be part of tests instead
  # of in official method
  sparsity_two = (sum(np.count_nonzero(layer==0) for layer in layers) /
      sum(layer.size for layer in layers))
  assert sparsity_one == sparsity_two

  return layers, {'shape': shape, 'paths': paths, 'connections_per_neuron': connections_per_neuron, 'sparsity': sparsity_one}


def kronecker_emr_network(radix_lists, B):
  """
  Creates a sparse network topology using the Kronecker/EMR method. First calls
  extended_mixed_radix_network(radix_lists). This network is then expanded via
  kronecker product to fill the fully connected structure defined by B. 

  Inputs:
  radix_lists - A list of lists of radices to be used for the [possibly extended]
      mixed radix system. The number of entries is the number of layers in the 
      network W that is output.
  B - a list of integers giving the number of neurons per layer of the
      superstructure into which the EMR network is being Kroneckered.

  Returns:
  layers - A list of numpy arrays where the i'th entry is the weight matrix W_i
      for the i'th layer.
  info   - A dict containing info about the layers with the following key/values:
      shape: a tuple containing the number of neurons for each layer. NOTE:
      this includes the input and output layers. as a result, 
      len(shape) = len(layers) + 1. That is, layers doesn't give the input weights.
      For the EMR method, the number of neurons is the same for each layer.
      paths: the number of possible paths between each input and output neuron.
      sparsity: the sparsity of the network (fraction of entries which are zero)
      connections_per_neuron: the average number of outgoing connections per neuron
      Note that
      sparsity = onnections_per_neuron * num_neurons / total_connections_possible

  """
  emr_layers, info = extended_mixed_radix_network(radix_lists)
  num_layers = len(emr_layers)
  emr_shape, emr_paths, emr_connections_per_neuron = info['shape'], info['paths'],\
      info['connections_per_neuron']
  shape = [emr_shape[i] * B[i] for i in range(len(shape))]
  # the first and last numbers in B do not add paths, but increase the
  # number of input and output neurons.
  paths *= np.prod(B[1:-1]) 

  # check valid input for B
  if len(B) - 1 != num_layers:
    raise  ValueError('Incorrect lengths of N, B parameters')

  # make the B graph to kronecker with emr_layers
  B_layers = [np.ones((B[i], B[i+1])) for i in range(len(B)-1)]

  expanded_layers = [np.kron(B_layer, emr_layer)
      for (B_layer, emr_layer) in zip(B_layers, emr_layers)]

  # calculate info statistics
  connections_per_neuron = (sum(shape[i]*connections_per_neuron*B[i+1]
    for i in range(num_layers)) / sum(shape[:-1]))
  total_neurons = np.sum(shape[:-1])
  total_connections_possible = sum(shape[i]*shape[i+1] for i in range(num_layers))
  sparsity_one = 1 - connections_per_neuron * total_neurons \
      / total_connections_possible
  sparsity_two = (sum(np.count_nonzero(layer==0) for layer in expanded_layers) \
      / sum(layer.size for layer in expanded_layers))
  assert sparsity_one == sparsity_two

  return expanded_layers, {'shape': shape, 'paths': paths, 'connections_per_neuron', connections_per_neuron, 'sparsity', sparsity}

      
if __name__ == '__main__':
  radix_lists = [[2,2]]
  B = [1,2,3]
#  layers = extended_mixed_radix_network(radix_lists)
  kron_network = kronecker_emr_network(radix_lists, B)
