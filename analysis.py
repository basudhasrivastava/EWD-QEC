import numpy as np
import random as rand
import pandas as pd

import copy
import collections

from numba import jit, prange
from src.toric_model import Toric_code
from src.util import Action
from src.RL import prediction_mod
from mcmc import *

def main(filename, prediction_args):
    size = 5
    reader = MCMCDataReader(filename, size)

    names = ['qubit_matrix', 'drl_correction_chain']
    tmp_dict = {name: None for name in names}
    data = []

    while reader.has_next():
        qubit_matrix, eq_distr = reader.next()
        current_toric = Toric_code(size)
        current_toric.qubit_matrix = qubit_matrix
        current_toric.syndrom('next_state')

        ret = prediction_mod(**prediction_args, toric=current_toric) # Anpassa efter Alexeis kod
        drl_success = ret[-2]
        drl_correction_chain = ret[-1]
        if not drl_success:
            tmp_dict['qubit_matrix'] = qubit_matrix
            tmp_dict['drl_correction_chain'] = drl_correction_chain
            data.append(tmp_dict)
        
    df = pd.DataFrame(data)
    df.to_pickle('test0.xy')


# Returns number of degenerate equivalence classes
def count_degenerate(eq_distr, rel_tol=0.1):
    sorted_distr = np.sort(eq_distr)[::-1]

    i = 0
    while sorted_distr[i+1] > sorted_distr[i] * (1 - rel_tol):
        i += 1
    
    return i + 1





if __name__ == '__main__':
    main()