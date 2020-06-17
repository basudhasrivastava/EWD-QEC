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


def single_temp(init_toric, p, max_iters, eps, burnin = 625, conv_criteria = 'error_based'):
    nbr_eq_class = 16
    ground_state = define_equivalence_class(init_toric.qubit_matrix)
    ladder = [] # list of chain objects
    nbr_errors_chain = np.zeros((16, max_iters))
    convergence_reached = np.zeros(16)
    mean_array = np.zeros(16)
    
    eq_array_translate = np.zeros(16)
    
    for i in range(nbr_eq_class):
        ladder.append(Chain(init_toric.system_size, p))
        ladder[i].toric = copy.deepcopy(init_toric)  # give all the same initial state
        ladder[i].toric.qubit_matrix = apply_logical_operator(ladder[i].toric.qubit_matrix, i) # apply different logical operator to each chain
        eq_array_translate[i] = define_equivalence_class(ladder[i].toric.qubit_matrix)
    print(eq_array_translate)
    for i in range(nbr_eq_class):
        since_burn = 0
        for j in range(max_iters):
            ladder[i].update_chain(1)
            
            if not convergence_reached[i] and j >= burnin:
                nbr_errors_chain[i ,j-burnin] = np.count_nonzero(ladder[i].toric.qubit_matrix)
                since_burn+=1
                if conv_criteria == 'error_based':
                    convergence_reached[i] = conv_crit_error_based(nbr_errors_chain[i, :j], since_burn, eps)
                    if convergence_reached[i] == 1: 
                        mean_array[i] = np.average(nbr_errors_chain[i ,:j])
                        print(j, 'convergence iterations')
                        break 
    print(ground_state, 'ground state')
    return mean_array, convergence_reached, eq_array_translate
            
def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):
        
        if binary[0] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)
        
        return qubit_matrix

def conv_crit_error_based(nbr_errors_chain, since_burn, eps):  # Konvergenskriterium 1 i papper
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    l = since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_chain[(3 * l // 4): l])

    # Compare averages
    error = abs(Average_Q2 - Average_Q4)

    if error < eps:
        return 1
    else:
        return 0
    
   
if __name__ == '__main__':
    init_toric = Toric_code(5)
    p_error = 0.15
    init_toric.generate_random_error(p_error)
    mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error, max_iters = 100000, eps = 0.000001, burnin = 10, conv_criteria = 'error_based') 
    print(eq_array_translate[np.argmin(mean_array)], 'guess')
    print(mean_array, convergence_reached)     
    
    
    
    
    
    