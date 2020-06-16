import numpy as np
import random as rand
import pandas as pd

import copy
import collections
import os

from numba import jit, prange
from src.toric_model import Toric_code
from src.util import Action
from NN import NN_11, NN_17, NN_11_mod
from tqdm import tqdm
from src.mcmc import *
from src.RL import *


def main(file_path, RL_args, prediction_args):
    size = 5

    names = ['qubit_matrix', 'drl_correction_chain']
    data = []

    rl = RL(**RL_args)
    rl.load_network(prediction_args['PATH'])
    reader = rl.mcmc_data_reader

    for i in tqdm(range(reader.get_capacity())):
        tmp_dict = {}
        qubit_matrix, eq_distr = reader.next()
        print(qubit_matrix)

        success, correction_list = rl.prediction_mod(**prediction_args, qubit_matrix=qubit_matrix) # Anpassa efter Alexeis kod
        drl_correction_chain = correction_list[-1] # last value in correction_list is complete correction chain

        drl_smart = (define_equivalence_class(drl_correction_chain) == np.argmax(eq_distr))
        if success and not drl_smart:
            tmp_dict['qubit_matrix'] = qubit_matrix
            tmp_dict['drl_correction_chain'] = drl_correction_chain
            data.append(tmp_dict)
        
    df = pd.DataFrame(data)
    df.to_pickle('drl_failures.xz')


# Returns number of degenerate equivalence classes
def count_degenerate(eq_distr, rel_tol=0.1):
    sorted_distr = np.sort(eq_distr)[::-1]

    i = 0
    while sorted_distr[i+1] > sorted_distr[i] * (1 - rel_tol):
        i += 1
    
    return i + 1


if __name__ == '__main__':
    size = 5
    p_error = 0.1
    file_path = './data/data_5x5_p_0.2.xz'
    prediction_args = {'prediction_list_p_error': [0.0], 'PATH': './network/Size_5_NN_11.pt'}
    RL_args = {'Network': NN_11, 'Network_name': 'Size_5_NN_11', 'system_size': size, 
        'p_error': p_error, 'replay_memory_capacity': 1e4, 'DATA_FILE_PATH': file_path}
    main(file_path, RL_args, prediction_args)