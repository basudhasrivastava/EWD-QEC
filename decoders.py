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
from operator import itemgetter

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

    ops = eq_to_ops(number ^ define_equivalence_class(qubit_matrix))

    for layer, op in enumerate(ops):
        apply_logical(qubit_matrix, operator=op, layer=layer, X_pos=0, Z_pos=0)

    return qubit_matrix

# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs
# vill göra detta men med mwpm? verkar finnas sätt att hitta "alla" kortaste, frågan är om man även kan hitta alla längre också
# https://stackoverflow.com/questions/58605904/finding-all-paths-in-weighted-graph-from-node-a-to-b-with-weight-of-k-or-lower
# i nuläget kommer "bra eq" att bli straffade eftersom att de inte kommer få chans att generera lika många unika kedjor --bör man sätta något tak? eller bara ta med de kortaste inom varje?


def single_temp_direct_sum(qubit_matrix, size, p, steps=20000):
    chain = Chain(size, p)  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    qubitlist = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]

    for i in range(16):
        chain.toric.qubit_matrix = apply_logical_operator(qubit_matrix, i)  # apply different logical operator to each chain
        # We start in a state with high entropy, therefore we let mcmc "settle down" before getting samples.
        current_class = define_equivalence_class(chain.toric.qubit_matrix)
        for _ in range(int(steps*0.8)):
            chain.update_chain(5)
        for _ in range(int(steps*0.2)):
            chain.update_chain(5)
            qubitlist[current_class][chain.toric.qubit_matrix.tostring()] = np.count_nonzero(chain.toric.qubit_matrix)

    # --------Determine EC-Distrubution--------
    eqdistr = np.zeros(16)
    beta = -log((p / 3) / (1-p))

    for i in range(16):
        for key in qubitlist[i]:
            eqdistr[i] += exp(-beta*qubitlist[i][key])

    return (np.divide(eqdistr, sum(eqdistr)) * 100).astype(np.uint8)


def single_temp_relative_count(qubit_matrix, size, p_error, p_sampling=None, steps=20000):
    p_sampling = p_sampling or p_error
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error
    
    Z_arr = np.zeros(16)
    max_length = 2 * size ** 2

    samples = int(0.9 * steps)

    init_eq = define_equivalence_class(qubit_matrix)
    ordered_ops = eq_to_ordered_ops(init_eq)

    chain = Chain(size, p_sampling)  # this p needs not be the same as p, as it is used to determine how we sample N(n)

    for eq in range(16):
        unique_lengths = {}
        len_counts = {}
        # List where first (last) element is stats of shortest (next shortest) length
        # n is length of chain. N is number of unique chains of this length
        short_stats = [{'n':max_length, 'N':0} for _ in range(2)]
        chain.toric.qubit_matrix = qubit_matrix
        # Apply logical operators to get qubit_matrix into equivalence class i
        for layer in range(2):
            chain.toric.qubit_matrix, _ = apply_logical(chain.toric.qubit_matrix, ordered_ops[eq][layer], layer)

        for _ in range(steps - samples):
            chain.update_chain(5)
        for step in range(samples):
            chain.update_chain(5)
            
            key = chain.toric.qubit_matrix.tostring()

            # Check if this error chain has already been seen
            if key in unique_lengths:
                # Increment counter for chains of this length
                len_counts[unique_lengths[key]] += 1
            
            # If this chain is new, add it to dictionary of unique chains
            else:
                # Calculate length of this chain
                length = np.count_nonzero(chain.toric.qubit_matrix)
                # Store number of observations and length of this chain
                unique_lengths[key] = length

                # Check if this length has been seen before
                if length in len_counts:
                    len_counts[unique_lengths[key]] += 1

                    # Otherwise, check if this chain is same length as previous shortest chain
                    if length == short_stats[0]['n']:
                        # Then increment counter of unique chains of shortest length
                        short_stats[0]['N'] += 1

                    # Otherwise, check if this chain same length as previous next shortest chain
                    elif length == short_stats[1]['n']:
                        # Then increment counter of unique chains of next shortest length
                        short_stats[1]['N'] += 1
                    
                else:
                    # Initiate counter for chains of this length
                    len_counts[unique_lengths[key]] = 1
                    # Check if this chain is shorter than prevous shortest chain
                    if length < short_stats[0]['n']:
                        # Then the previous shortest length is the new next shortest
                        short_stats[1] = short_stats[0]
                        # And the current length is the new shortest
                        short_stats[0] = {'n':length, 'N':1}

                    # Otherwise, check if this chain is shorter than previous next shortest chain
                    elif length < short_stats[1]['n']:
                        # Then reset stats of next shortest chain
                        short_stats[1] = {'n':length, 'N':1}

        # Dict to hold the total occurences and unique chains of each observed length
        shortest = short_stats[0]['n']
        shortest_count = short_stats[0]['N']
        next_shortest = short_stats[1]['n']
        next_shortest_count = short_stats[1]['N']

        shortest_fraction = shortest_count / len_counts[shortest]
        next_shortest_fraction = next_shortest_count / len_counts[next_shortest]
        mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))

        Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts.items()]) * mean_fraction

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
    size = 7
    steps = 10000 * int(1 + (size / 5) ** 4)
    #reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.19
    p_sampling = 0.19
    init_toric = Toric_code(size)
    tries = 2
    distrs = np.zeros((tries, 16), dtype=int)
    mean_tvd = 0.0
    for i in range(10):
        init_toric.generate_random_error(p_error)
        init_qubit = np.copy(init_toric.qubit_matrix)
        #init_qubit, mcmc_distr = reader.next()
        #print(init_qubit)
        print('################ Chain', i+1 , '###################')
        #print('MCMC distr:', mcmc_distr)
        for i in range(tries):
            distrs[i] = single_temp_relative_count(init_qubit, size=size, p_error=p_error, p_sampling=p_sampling, steps=steps)
            print('Try', i+1, ':', distrs[i])
        
        tvd = sum(abs(distrs[1]-distrs[0]))
        mean_tvd += tvd
        print('TVD:', tvd)
    print('Mean TVD:', mean_tvd / 10)
        
        #print('STRC distribution 1:', distr1)
        #print('STRC distribution 2:', distr2)