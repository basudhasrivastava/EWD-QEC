import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import from_numpy
from collections import namedtuple
import pandas as pd


Action = namedtuple('Action', ['position', 'action'])

Perspective = namedtuple('Perspective', ['perspective', 'position'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])


def conv_to_fully_connected(input_size, filter_size, padding, stride):
    return (input_size - filter_size + 2 * padding)/ stride + 1


def pad_circular(x, pad):
    x = torch.cat([x, x[:,:,:,0:pad]], dim=3)
    x = torch.cat([x, x[:,:, 0:pad,:]], dim=2)
    x = torch.cat([x[:,:,:,-2 * pad:-pad], x], dim=3)
    x = torch.cat([x[:,:, -2 * pad:-pad,:], x], dim=2)
    return x


def incremental_mean(x, mu, N):
    return mu + (x - mu) / (N)


def convert_from_np_to_tensor(tensor):
    tensor = from_numpy(tensor)
    tensor = tensor.type('torch.Tensor')
    return tensor

def getMCMCstats(qubit_matrix_in, size, p_error, Nc=19, steps=1000000, crit='error_based'):
    """Get statistics about distribution of error chains given a syndrom (error chain) using MCMC-sampling."""
    init_toric = Toric_code(size)
    
    # define error
    init_toric.qubit_matrix = qubit_matrix_in
    init_toric.syndrom('next_state')

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    # Start in random eq-class
    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    distr, count, qubitlist = parallel_tempering_plus(init_toric, Nc, p=p_error, steps=steps, iters=10, conv_criteria=crit)
    print(distr)
    unique_elements, unique_counts = np.unique(qubitlist, axis=0, return_counts=True)
    print('Number of unique elements: ', len(unique_elements))

    shortest = inf
    for i in range(len(qubitlist)):
        nb = np.count_nonzero(qubitlist[i])
        if nb < shortest:
            shortest = nb

    # save all qubits to df_all
    df = pd.DataFrame({"qubit":[], "nbr_err":[], "nbr_occ":[], "eq_class":[]})
    df = pd.concat((pd.DataFrame({"qubit":[unique_elements[i]], "nbr_err":[np.count_nonzero(unique_elements[i])], "nbr_occ":[unique_counts[i]], "eq_class": define_equivalence_class(unique_elements[i])}) for i in range(len(unique_elements))),
            ignore_index=True)
    
    for i in range(7):
        print(shortest+i)
        print(df.loc[df['nbr_err'] == shortest + i])
        for j in range(16):
            nbr_comb = len(df.loc[df['nbr_err'] == shortest + i].loc[df['eq_class'] == j])
            if nbr_comb > 0:
                print('class ', j, '\t\t', nbr_comb)