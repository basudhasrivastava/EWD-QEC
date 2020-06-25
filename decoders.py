import numpy as np
import random as rand
import copy
import collections


from numba import jit, prange
from src.toric_model import Toric_code
from src.util import *
from src.mcmc import *
import pandas as pd

from math import log, exp

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
        for j in range(max_iters):
            ladder[i].update_chain(1)
            nbr_errors_chain[i ,j] = np.count_nonzero(ladder[i].toric.qubit_matrix)
            if not convergence_reached[i] and j >= burnin:
                if conv_criteria == 'error_based':
                    convergence_reached[i] = conv_crit_error_based(nbr_errors_chain[i, :j], j, eps)
                    if convergence_reached[i] == 1: 
                        mean_array[i] = np.average(nbr_errors_chain[i ,:j])
                        print(j, 'convergence iterations')
                        break 
    print(ground_state, 'ground state')
    return mean_array, convergence_reached, eq_array_translate


def conv_crit_error_based(nbr_errors_chain, l, eps):  # Konvergenskriterium 1 i papper
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_chain[(3 * l // 4): l])

    # Compare averages
    error = abs(Average_Q2 - Average_Q4)

    if error < eps:
        return 1
    else:
        return 0


def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):
        
        if binary[0] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1': qubit_matrix, _  = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)
        
        return qubit_matrix


def single_temp_mcmc(qubit_matrix, size, p, steps=20000):
    chain = Chain(size, p)  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    samples = int(0.9 * steps)
    qubit_list = np.zeros((16, samples, 2, size, size))

    init_eq = define_equivalence_class(qubit_matrix)
    ordered_ops = eq_to_ordered_ops(init_eq)

    for eq in range(16):
        chain.toric.qubit_matrix = qubit_matrix
        # Apply logical operators to get qubit_matrix into equivalence class i
        for layer in range(2):
            chain.toric.qubit_matrix, _ = apply_logical(chain.toric.qubit_matrix, ordered_ops[eq][layer], layer)

        for _ in range(steps - samples):
            chain.update_chain(5)
        for step in range(samples):
            chain.update_chain(5)
            qubit_list[eq][step] = chain.toric.qubit_matrix

    return qubit_list
# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs
# vill göra detta men med mwpm? verkar finnas sätt att hitta "alla" kortaste, frågan är om man även kan hitta alla längre också
# https://stackoverflow.com/questions/58605904/finding-all-paths-in-weighted-graph-from-node-a-to-b-with-weight-of-k-or-lower
# i nuläget kommer "bra eq" att bli straffade eftersom att de inte kommer få chans att generera lika många unika kedjor --bör man sätta något tak? eller bara ta med de kortaste inom varje?


def single_temp_direct_sum(qubit_matrix, size, p, steps=20000):
    chain = Chain(size, p)  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    qubitlist = []

    for i in range(16):
        chain.toric.qubit_matrix = apply_logical_operator(qubit_matrix, i)  # apply different logical operator to each chain
        # We start in a state with high entropy, therefore we let mcmc "settle down" before getting samples.
        for _ in range(int(steps*0.1)):
            chain.update_chain(5)
        for _ in range(int(steps*0.9)):
            chain.update_chain(5)
            qubitlist.append(chain.toric.qubit_matrix)

    # Only consider unique elements
    qubitlist = np.unique(qubitlist, axis=0)

    # --------Determine EC-Distrubution--------
    eqdistr = np.zeros(16)
    beta = -log((p / 3) / (1-p))

    for i in range(len(qubitlist)):
        eq = define_equivalence_class(qubitlist[i])
        eqdistr[eq] += exp(-beta*np.count_nonzero(qubitlist[i]))

    return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)


def single_temp_relative_count(qubit_matrix, size, p, steps=20000):
    qubit_list = single_temp_mcmc(qubit_matrix, size, p, steps)
    
    beta = -log((p / 3) / (1 - p))
    Z_arr = np.zeros(16)

    for eq in range(16):
        # Get array of unique error chains and their occurences
        unique_chains, unique_counts = np.unique(qubit_list[eq], return_counts=True, axis=0)
        # Keep track of the unique chains in each class

        # Keep track of the shortest and next shortest chain lengths
        shortest = 2 * size ** 2
        next_shortest = shortest
        # Dict to hold the total occurences and unique chains of each observed length
        Nm_n = {}

        # Iterate through the unique chains
        for i, chain in enumerate(unique_chains):
            # Get the current chain length, store it
            length = np.count_nonzero(chain)
            # Update shortest and next shortest lengths
            if length < shortest:
                next_shortest = shortest
                shortest = length
            elif length < next_shortest:
                next_shortest = length

            # Update the occurences of chain lengths
            if length in Nm_n:
                Nm_n[length] += np.array([1, unique_counts[i]])
            else:
                Nm_n[length] = np.array([1, unique_counts[i]])

        shortest_counts = Nm_n[shortest]
        shortest_fraction = shortest_counts[0] / shortest_counts[1]
        next_shortest_counts = Nm_n[next_shortest]
        next_shortest_fraction = next_shortest_counts[0] / next_shortest_counts[1]
        mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta * (next_shortest - shortest))) * exp(-beta * shortest)

        Z_e = sum(Nm_n.values())[1] * mean_fraction

        Z_arr[eq] = Z_e

    return (Z_arr / np.sum(Z_arr) * 100).astype(dtype=int)


if __name__ == '__main__':
    '''
    init_toric = Toric_code(5)
    p_error = 0.1
    init_toric.generate_random_error(p_error)
    mean_array, convergence_reached, eq_array_translate = single_temp(init_toric, p = p_error, max_iters = 1000000, eps = 0.00001, burnin = 100000, conv_criteria = 'error_based') 
    print(eq_array_translate[np.argmin(mean_array)], 'guess')
    print(convergence_reached)
    '''
    size = 5
    reader = MCMCDataReader('data/data_5x5_p_0.15.xz', size)
    p_error = 0.15
    for i in range(10):
        init_qubit, mcmc_distr = reader.next()
        print(init_qubit)
        print('MCMC distribution:', mcmc_distr)
        distr = single_temp_relative_count(init_qubit, size=5, p=p_error, steps=20000)
        print('STRC distribution:', distr)
    