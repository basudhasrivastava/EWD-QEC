import numpy as np
import random as rand
import copy
import collections


from numba import jit, prange
from src.toric_model import Toric_code
from src.util import Action
from src.mcmc import *
from src.mcmc import Chain
import pandas as pd
from math import *

# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs

#@profile
def single_temp_direct_sum(qubit_matrix, size, p, steps):
    init_toric = Toric_code(size)
    init_toric.qubit_matrix = qubit_matrix
    nbr_eq_class = 16
    ladder = [] # list of chain objects

    # save N_E(n) på något sätt?
    qubitlist = [] # make these into sets!

    for i in range(nbr_eq_class):
        ladder.append(Chain(init_toric.system_size, p))
        ladder[i].toric = copy.deepcopy(init_toric)  # give all the same initial state
        ladder[i].toric.qubit_matrix = apply_logical_operator(ladder[i].toric.qubit_matrix, i) # apply different logical operator to each chain
            # here we start in a high entropy state for most eqs, which is not desired as it increases time to find smaller solutions.
    for i in range(nbr_eq_class):
        for _ in range(steps):
            ladder[i].update_chain(1)
            qubitlist.append(ladder[i].toric.qubit_matrix)
        print(len(qubitlist))
    qubitlist = np.unique(qubitlist, axis = 0)
    #######--------Determine EQC--------########

    eqdistr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    beta = -log(p/3/(1-p))

    for val in qubitlist:
        qubit_matrix = np.array(list(val))
        eq = define_equivalence_class(qubit_matrix)
        eqdistr[eq] += exp(-beta*np.count_nonzero(qubit_matrix))
        
    return [int(x * 100 / sum(eqdistr)) for x in eqdistr]
            
def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):
        
        if binary[0] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)
        
        return qubit_matrix


if __name__ == '__main__':
    init_toric = Toric_code(5)
    p_error = 0.15
    init_toric.generate_random_error(p_error)
    print(single_temp_direct_sum(init_toric.qubit_matrix, 5, p = p_error, steps = 100000))
    print(init_toric.qubit_matrix)
    #distr, count, qubitlist = parallel_tempering_plus(init_toric, 19, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
    #print(distr)
