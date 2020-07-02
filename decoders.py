import numpy as np
import random as rand
import copy
import collections
import time

from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.util import *
from src.mcmc import *
import pandas as pd
import time

from math import log, exp
from operator import itemgetter
from multiprocessing import Pool

def single_temp(init_code, p, max_iters):
    nbr_eq_classes = init_code.nbr_eq_classes
    ground_state = init_code.define_equivalence_class()
    ladder = [] # list of chain objects
    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    mean_array = np.zeros(nbr_eq_classes, dtype=float)

    for eq in range(nbr_eq_classes):
        ladder.append(Chain(init_code.system_size, p, copy.deepcopy(init_code)))
        ladder[eq].code.qubit_matrix = ladder[eq].code.to_class(eq) # apply different logical operator to each chain

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            ladder[eq].update_chain(5)
            nbr_errors_chain[eq ,j] = ladder[eq].code.count_errors()
            if j == max_iters-1:
                mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])

    return mean_array


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


def STDC_droplet(input_data_tuple):
    # All unique chains will be saved in samples
    samples = {}
    chain, steps = input_data_tuple

    # Start in high energy state
    chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Do the metropolis steps and add to samples if new chains are found
    for _ in range(int(steps)):
        chain.update_chain(5)
        key = chain.code.qubit_matrix.astype(np.uint8).tostring()
        if key not in samples:
            samples[key] = chain.code.count_errors()

    return samples


def STDC(init_code, size, p_error, p_sampling=None, droplets=10, steps=20000):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    # Create chain with p_sampling, this is allowed since N(n) is independet of p.
    chain = Chain(size, p_sampling, copy.deepcopy(init_code))

    # this is either 4 or 16, depending on what type of code is used.
    nbr_eq_classes = init_code.nbr_eq_classes

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    # error-model
    beta = -log((p_error / 3) / (1 - p_error))

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain.code.qubit_matrix = init_code.to_class(eq)

        with Pool(droplets) as pool:
            output = pool.map(STDC_droplet, [(copy.deepcopy(chain), steps) for _ in range(droplets)])
            for j in range(droplets):
                qubitlist.update(output[j])

        # compute Z_E        
        for key in qubitlist:
            eqdistr[eq] += exp(-beta * qubitlist[key])
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100)


def STRC_droplet(input_data_tuple):
    chain, steps, max_length, eq = input_data_tuple
    unique_lengths = {}
    len_counts = {}

    # List of unique shortest and next shortets chains
    short_unique = [{} for _ in range(2)]

    # Variables to easily keep track of the length of chains in short_unique
    shortest = max_length
    next_shortest = max_length
    
    # Apply random stabilizers to start in high temperature state
    chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()
    # Apply logical operators to get qubit_matrix into equivalence class eq
    chain.code.qubit_matrix = chain.code.to_class(eq)

    # Generate chains
    for step in range(steps):
        # Do metropolis sampling
        chain.update_chain(5)

        # Convert the current qubit matrix to string for hashing
        key = chain.code.qubit_matrix.tostring()

        # Check if this error chain has already been seen by comparing hashes
        if key in unique_lengths:
            # Increment counter for chains of this length
            len_counts[unique_lengths[key]] += 1

        # If this chain is new, add it to dictionary of unique chains
        else:
            # Calculate length of this chain
            length = chain.code.count_errors()
            # Store number of observations and length of this chain
            unique_lengths[key] = length

            # Check if this length has been seen before
            if length in len_counts:
                len_counts[unique_lengths[key]] += 1

                # Check if this chain is same length as previous shortest chain
                if length == shortest:
                    # Then add it to the set of seen short chains
                    short_unique[0][key] = length

                # Otherwise, check if this chain same length as previous next shortest chain
                elif length == next_shortest:
                    # Then add it to the set of seen next shortest chains
                    short_unique[1][key] = length

            else:
                # Initiate counter for chains of this length
                len_counts[unique_lengths[key]] = 1
                # Check if this chain is shorter than prevous shortest chain
                if length < shortest:
                    # Then the previous shortest length is the new next shortest
                    next_shortest = shortest
                    shortest = length
                    
                    # Clear next shortest set and set i equal to shortest
                    short_unique[1].clear()
                    short_unique[1].update(short_unique[0])
                    # And the current length is the new shortest
                    short_unique[0].clear()
                    short_unique[0][key] = length
                
                # Otherwise, check if this chain is shorter than previous next shortest chain
                elif length < next_shortest:
                    # Then reset stats of next shortest chain
                    next_shortest = length

                    # Clear and update next shortest set
                    short_unique[1].clear()
                    short_unique[1][key] = length

    return unique_lengths, len_counts, short_unique

 
def STRC(init_code, size, p_error, p_sampling=None, droplets=10, steps=20000):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    # Create chain with p_sampling, this is allowed since N(n) is independet of p.
    chain = Chain(size, p_sampling, copy.deepcopy(init_code))

    # Either 4 or 16, depending on type of code
    nbr_eq_classes = init_code.nbr_eq_classes

    # error model
    beta_error = -log((p_error / 3) / (1 - p_error))
    beta_sampling = -log((p_sampling / 3) / (1 - p_sampling))
    d_beta = beta_sampling - beta_error

    # Array to hold the boltzmann factors for every class
    Z_arr = np.zeros(nbr_eq_classes)

    # Largest possible chain length
    max_length = 2 * size ** 2

    # Iterate through equivalence classes
    for eq in range(nbr_eq_classes):
        # Start parallel processes with droplets.
        with Pool(droplets) as pool:
            output = pool.map(STRC_droplet, [(copy.deepcopy(chain), steps, max_length, eq) for _ in range(droplets)])

        # We need to combine the results from all raindrops
        unique_lengths = {}
        len_counts = {}
        short_unique = [{} for _ in range(2)]

        shortest = max_length
        next_shortest = max_length

        # Find shortest and next shortest length found by any chain
        for i in range(droplets):
            _,_,data = output[i]
            if rand.choice(list(data[0].values())) < shortest:
                next_shortest = shortest
                shortest = rand.choice(list(data[0].values()))
            if rand.choice(list(data[1].values())) < next_shortest:
                next_shortest = rand.choice(list(data[1].values()))
        
        # Add data from each droplet to the combined dataset
        for i in range(droplets):
            # Unpack results
            unique_lengths_i, len_counts_i, short_unique_i = output[i]
            
            # Combine unique lengths ( not really needed? )
            unique_lengths.update(unique_lengths_i)

            # Combine len_counts
            for key in len_counts_i:
                if key in len_counts:
                    len_counts[key] += len_counts_i[key]
                else:
                    len_counts[key] = len_counts_i[key]
            
            # Combine the sets of shortest and next shortest chains
            shortest_i = rand.choice(list(short_unique_i[0].values()))
            next_shortest_i = rand.choice(list(short_unique_i[1].values()))

            if shortest_i == shortest:
                short_unique[0].update(short_unique_i[0])
            if shortest_i == next_shortest:
                short_unique[1].update(short_unique_i[0])
            if next_shortest_i == next_shortest:
                short_unique[1].update(short_unique_i[1])

        # Partial result needed for boltzmann factor
        shortest_count = len(short_unique[0])
        shortest_fraction = shortest_count / len_counts[shortest]

        next_shortest_count = len(short_unique[1])
        
        # Handle rare cases where only one chain length is observed
        if next_shortest != max_length:
            next_shortest_fraction = next_shortest_count / len_counts[next_shortest]
            mean_fraction = 0.5 * (shortest_fraction + next_shortest_fraction * exp(-beta_sampling * (next_shortest - shortest)))
        
        else:
            mean_fraction = shortest_fraction

        # Calculate boltzmann factor from observed chain lengths
        Z_e = sum([m * exp(-beta_sampling * shortest + d_beta * l) for l, m in len_counts.items()]) * mean_fraction
        Z_arr[eq] = Z_e

    # Use boltzmann factors as relative probabilities and normalize distribution
    return (Z_arr / np.sum(Z_arr) * 100)


if __name__ == '__main__':
    t0 = time.time()
    size =  11
    steps = 10000 * int(1 + (size / 5) ** 4)
    #reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.05
    p_sampling = 0.05
    init_code = Planar_code(size)
    tries = 2
    distrs = np.zeros((tries, init_code.nbr_eq_classes))
    mean_tvd = 0.0
    for i in range(10):
        init_code.generate_random_error(p_error)
        ground_state = init_code.define_equivalence_class()
        init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
        init_qubit = np.copy(init_code.qubit_matrix)

        print('################ Chain', i+1 , '###################')

        for i in range(tries):
            #print('Try', i+1, ':', distrs[i], 'most_likeley_eq', np.argmax(distrs[i]), 'ground state:', ground_state)

            #v1, most_likely_eq, convergece = single_temp(init_code, p=p_error, max_iters=steps, eps=0.005, conv_criteria = None)
            #print('Try single_temp', i+1, ':', v1, 'most_likely_eq', most_likely_eq, 'ground state:', ground_state, 'convergence:', convergece, time.time()-t0)
            t0 = time.time()
            distrs[i] = STDC(copy.deepcopy(init_code), size=size, p_error=p_error, steps=steps)
            print('Try STDC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)
            t0 = time.time()
            distrs[i] = STRC(copy.deepcopy(init_code), size=size, p_error=p_error, p_sampling=p_sampling, steps=steps)
            print('Try STRC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)

            t0 = time.time()

        tvd = sum(abs(distrs[1]-distrs[0]))
        mean_tvd += tvd
        print('TVD:', tvd)
    print('Mean TVD:', mean_tvd / 10)

        #print('STRC distribution 1:', distr1)
        #print('STRC distribution 2:', distr2)
