from __future__ import division
import numpy as np
import fractions

def emr_net(radix_lists):
    """ Creates an X-net sparse network topology from the input parameters
    using Ryan's Extended Mixed-Radix method. See the documentation for a full
    explanation of what is going on.

    Inputs:

    radix_lists - A list of lists of radices to be used for the [possibly
    extended] mixed radix system. The number of entries is the number of layers
    in the network W that is output.

    Returns:

    layers - A list of numpy arrays where the i'th entry is the weight matrix
    W_i for the i'th layer.

    info - A dict containing info with the following key/values, where the key
    is the string of the info type:

        shape: a tuple containing the number of neurons for each layer. NOTE:
        this includes the input and output layers. as a result, len(shape) =
        len(layers) + 1. That is, layers doesn't give the input weights.  For
        the EMR method, the number of neurons is the same for each layer.

        paths: the number of possible paths between each input and output
        neuron.

        sp: a list containing the sparsity of each layer

        avg_sp: the average sparsity of the whole network

        conns: a list containing the number of connections in each layer.

        cpn: a list containing the average conns per neuron for each
        layer. This is simply the radix used for that layer.

        avg_cpn: the average conns per neuron of the whole network. Note
        that sp = 1 - (cpn * num_neurons / total_conns_possible).

    """

    # print('radix lists: ' + str(radix_lists))
    num_layers = sum(len(radix_list) for radix_list in radix_lists)
    # this is the number of neurons per layer.
    num_neurons = np.prod(radix_lists[0])

    # for all but last radix list, product of radices must equal num_neurons
    if not np.all( [num_neurons == np.prod(radix_list)
            for radix_list in radix_lists[:-1]]):
        raise ValueError('Product of radices for each radix list must equal'
            + 'number of neurons from first radix list for all but last radix'
            + 'list')
    # for last radix list, product of radices must divide num_neurons
    if num_neurons % np.prod(radix_lists[-1]) != 0:
        raise ValueError('Product of radices for last radix list must divide'
             + 'number of neurons from first radix list')

    # the actual math part
    # The N x N identity matrix, used for constructing permutation matrices
    I = np.identity(num_neurons)

    # calculate info
    shape = tuple([num_neurons] * (num_layers + 1))
    # print('shape: ' + str(shape))
    paths  = np.prod([np.prod(radix_list) for radix_list in radix_lists[1:]])
    # print('paths: ' + str(paths))
    flattened_radices = [radix for radix_list in radix_lists
        for radix in radix_list]
    conns = [flattened_radices[i] * num_neurons
        for i in range(num_layers)]
    cpn = [radix for radix in flattened_radices]
    avg_cpn = np.average(cpn)
    # print('cpn: ' + str(cpn))
    # print('avg_cpn: ' + str(avg_cpn))
    sp = [1 - flattened_radices[i] / num_neurons for i in range(num_layers)]
    # print('sp: ' + str(sp))
    avg_sp = np.average(sp)
    # print('avg_sp: ' + str(avg_sp))

    layers = [] # the output list containing W_i's
    # make layers
    for radix_list in radix_lists:
        place_value = 1
        for radix in radix_list:
            layer = np.sum( [np.roll(I, -j * place_value, axis=1)
                for j in range(radix)], axis=0)
            layers.append(layer)
            place_value *= radix

    # sp calculated manually (much slower, should be part of tests
    # instead of in official method
    sp_two = [np.count_nonzero(layer==0) / layer.size for layer in layers]
    # print('sp_two: ' + str(sp_two))
    avg_sp_two = np.average(sp_two)
    # print('avg_sp_two: ' + str(avg_sp_two))
    assert sp == sp_two
    assert avg_sp == avg_sp_two

    return layers, {'shape': shape, 'paths': paths, 'conns': conns,
        'cpn': cpn, 'avg_cpn': avg_cpn, 'sp': sp, 'avg_sp': avg_sp}


def kemr_net(radix_lists, B):
    """ Creates a sparse network topology using the Kronecker/EMR method. First
    calls extended_mixed_radix_network(radix_lists). This network is then
    expanded via kronecker product to fill the fully connected structure
    defined by B.

    Inputs:

    radix_lists - A list of lists of radices to be used for the [possibly
    extended] mixed radix system. The number of entries is the number of layers
    in the network W that is output.

    B - a list of integers giving the number of neurons per layer of the
    superstructure into which the EMR network is being Kroneckered.

    Returns:

    layers - A list of numpy arrays where the i'th entry is the weight matrix
    W_i for the i'th layer.

    info - A dict containing info with the following key/values, where the key
    is the string of the info type:

        shape: a tuple containing the number of neurons for each layer. NOTE:
        this includes the input and output layers. as a result, len(shape) =
        len(layers) + 1. That is, layers doesn't give the input weights. For
        the K/emr method, the shape of layer i is emr_shape[i] * B[i], where
        emr_shape[i] is the shape of layer i in from the emr graph.

        paths: the number of possible paths between each input and output
        neuron.

        sp: a list containing the sparsity of each layer.

        avg_sp: the average sparsity of the whole network.

        conns: a list containing the number of conns in each layer.

        cpn: a list containing the average conns per neuron of each
        layer.

        avg_cpn: the average conns per neuron of the whole network. Note
        that sp = 1 - (cpn * num_neurons / total_conns_possible). Also
        note that for individual layers this is simply the radix used for that
        layer.

    """
    # print('radix lists: ' + str(radix_lists))
    # print('B: ' + str(B))

    emr_layers, info = emr_net(radix_lists)
    num_layers = len(emr_layers)
    emr_shape, emr_paths, emr_conns, emr_cpn, emr_sp = (info['shape'],
        info['paths'], info['conns'], info['cpn'], info['sp'])
    shape = [emr_shape[i] * B[i] for i in range(len(emr_shape))]
    # print('shape: ' + str(shape))
    # the first and last numbers in B do not add paths, but increase the
    # number of input and output neurons.
    paths = emr_paths*np.prod(B[1:-1])

    # check valid input for B
    if len(B) - 1 != num_layers:
        raise  ValueError('Incorrect lengths of N, B parameters; len(B) should'
            + 'be one more than num_layers')

    # make the B graph to kronecker with emr_layers
    B_layers = [np.ones((B[i], B[i+1])) for i in range(len(B)-1)]

    expanded_layers = [np.kron(B_layer, emr_layer)
        for (B_layer, emr_layer) in zip(B_layers, emr_layers)]

    assert all(expanded_layers[i].shape[0] == shape[i]
        for i in range(len(expanded_layers)))
    assert expanded_layers[-1].shape[1] == shape[-1]

    # calculate info statistics
    conns = [emr_conns[i]*B[i]*B[i+1] for i in range(num_layers)]
    # print('conns: ' + str(conns))
    cpn = [B[i+1] * emr_cpn[i] for i in range(num_layers)]
    avg_cpn = sum(cpn[i] * shape[i]
        for i in range(num_layers)) / sum(shape[:-1])
    # print('cpn: ' + str(cpn))
    # print('avg_cpn: ' + str(avg_cpn))
    sp = [1 - (cpn[i] / shape[i+1]) for i in range(num_layers)]
    # print('sp: ' + str(sp))
    avg_sp = (sum(sp[i] * shape[i] for i in range(num_layers))
        / sum(shape[:-1]))
    # print('avg_sp: ' + str(avg_sp))
    # note: avg_sp_2 takes a long time to calculate and should be removed
    # for "production" or time-sensitive use cases
    sp2 = [np.count_nonzero(layer==0) / layer.size
        for layer in expanded_layers]
    # print('sp2: ' + str(sp2))
    avg_sp_2 = (sum(sp2[i] * shape[i] for i in range(num_layers))
        / sum(shape[:-1]))
    # print('avg_sp_2: ' + str(avg_sp_2))
    assert sp == sp2


    return expanded_layers, {'shape': shape, 'paths': paths, 'conns': conns,
        'cpn': cpn, 'avg_cpn': avg_cpn, 'sp': sp, 'avg_sp': avg_sp}


if __name__ == '__main__':
    for b in range(3, 4):
        N = [[10, 10]]
        B = [8, b, 1]
        layers, info = kemr_net(N, B)
        print(str(info))
