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
from src.mwpm import class_sorted_mwpm
import pandas as pd
import time

from math import log, exp
from operator import itemgetter
from multiprocessing import Pool

# Original MCMC Parallel tempering method as descibed in high threshold paper
# Parameters also adapted from that paper.
# steps has an upper limit on 50 000 000, which should not be met during operation
def PTEQ(init_code, p, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    # System size is determined from init_code
    size = init_code.system_size

    # either 4 or 16 depending on choice of code topology
    nbr_eq_classes = init_code.nbr_eq_classes

    # If not specified, use size as per paper
    Nc = Nc or size

    # Warn about incorrect parameter inputs
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')

    ladder = []  # ladder to store all parallel chains with diffrent temperatures
    p_end = 0.75  # p at top chain is 0.75 (I,X,Y,Z equally likely)

    # initialize variables
    tops0 = 0
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)

    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)  # list of class counts after burn in

    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0

    # Convergence flag
    convergence_reached = False

    # Initialize all chains in ladder with same state but different temperatures.
    for i in range(Nc):
        p_i = p + ((p_end - p) / (Nc - 1)) * i # Temperature (in p) for chain i
        ladder.append(Chain(size, p_i))
        ladder[i].code= copy.deepcopy(init_code)  # give all the same initial state

    # Set probability of application of logical operator in top chain
    ladder[Nc - 1].p_logical = 0.5

    # Main loop that runs until convergence or max steps (steps) are reached
    for j in range(steps):
        # Run Metropolis steps for each chain in ladder
        for i in range(Nc):
            ladder[i].update_chain(iters)

        # Flag are used to know what chains originating in the top layer of the ladder has found their way down
        # The top chain always generates chains with flag "1". Once such a chain reaches the bottom the flag is
        # reset to 0
        ladder[-1].flag = 1

        # current_eq attempt flips of chains from the top down
        for i in reversed(range(Nc - 1)):
            if r_flip(ladder[i].code.qubit_matrix, ladder[i].p, ladder[i + 1].code.qubit_matrix, ladder[i + 1].p):
                ladder[i].code, ladder[i + 1].code = ladder[i + 1].code, ladder[i].code
                ladder[i].flag, ladder[i + 1].flag = ladder[i + 1].flag, ladder[i].flag
        
        # Update bottom chain flag and add to tops0
        if ladder[0].flag == 1:
            tops0 += 1
            ladder[0].flag = 0

        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder[0].code.define_equivalence_class()

        # Start saving stats once burn-in period is over
        if tops0 >= tops_burn:
            since_burn = j - resulting_burn_in

            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder[0].code.count_errors()
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1

        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = tops0
    
    # print warining if max nbr steps are reached before convergence
    if j + 1 == steps and conv_criteria == 'error_based':
        print('\n\nWARNING: PTEQ hit maxnbr steps before convergence:\t', j + 1, '\n\n')

    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)


@njit(cache=True) # r_flip calculates the quotient called r_flip in paper
def r_flip(qubit_lo, p_lo, qubit_hi, p_hi):
    ne_lo = 0
    ne_hi = 0
    for i in range(2):
        for j in range(qubit_lo.shape[1]):
            for k in range(qubit_lo.shape[1]):
                if qubit_lo[i, j, k] != 0:
                    ne_lo += 1
                if qubit_hi[i, j, k] != 0:
                    ne_hi += 1
    # compute eqn (5) in high threshold paper
    if rand.random() < ((p_lo / p_hi) * ((1 - p_hi) / (1 - p_lo))) ** (ne_hi - ne_lo):
        return True
    return False


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based_PT(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    l = since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_bottom_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_bottom_chain[(3 * l // 4): l])

    # Compare averages
    error = abs(Average_Q2 - Average_Q4)
    if error < eps:
        return True, tops_accepted >= SEQ
    else:
        return False, False


def single_temp(init_code, p, max_iters):
    # check if init_code is provided as a list of inits for different classes
    if type(init_code) == list:
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init per class is provided
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # initiate ladder
        ladder = [Chain(code.system_size, p, copy.deepcopy(code)) for code in init_code]
    
    # if init_code is a single code, inits for every class have to be generated
    else:
        nbr_eq_classes = init_code.nbr_eq_classes
        ladder = [None] * nbr_eq_classes # list of chain objects
        for eq in range(nbr_eq_classes):
            ladder[eq] = Chain(init_code.system_size, p, copy.deepcopy(init_code))
            ladder[eq].code.qubit_matrix = ladder[eq].code.to_class(eq) # apply different logical operator to each chain

    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    mean_array = np.zeros(nbr_eq_classes, dtype=float)

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            ladder[eq].update_chain_fast(5)
            nbr_errors_chain[eq ,j] = ladder[eq].code.count_errors()
            if j == max_iters-1:
                mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])

    return mean_array.round(decimals=2)


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
    chain, steps, randomize = input_data_tuple

    # Start in high energy state
    if randomize:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Do the metropolis steps and add to samples if new chains are found
    for _ in range(int(steps)):
        chain.update_chain_fast(5)
        key = chain.code.qubit_matrix.astype(np.uint8).tobytes()
        if key not in samples:
            samples[hash(key)] = chain.code.count_errors()
    return samples


def STDC(init_code, size, p_error, p_sampling=None, droplets=10, steps=20000):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [Chain(size, p_sampling, copy.deepcopy(code)) for code in init_code]
        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain(size, p_sampling, copy.deepcopy(init_code))
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)
        # apply uniform stabilizers, i.e. rain
        randomize = True

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    # error-model
    beta = -log((p_error / 3) / (1 - p_error))

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        if droplets == 1:
            qubitlist = STDC_droplet((copy.deepcopy(chain), steps, randomize))
        else:
            with Pool(droplets) as pool:
                output = pool.map(STDC_droplet, [(copy.deepcopy(chain), steps, True) for _ in range(droplets-1)] + [(copy.deepcopy(chain), steps, randomize)])
                for j in range(droplets):
                    qubitlist.update(output[j])

        # compute Z_E        
        for key in qubitlist:
            eqdistr[eq] += exp(-beta * qubitlist[key])
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100)


def STRC_droplet(input_data_tuple):
    chain, steps, max_length, eq, randomize = input_data_tuple
    unique_lengths = {}
    len_counts = {}

    # List of unique shortest and next shortets chains
    short_unique = [{} for _ in range(2)]
    short_unique[0]['temp'] = max_length
    short_unique[1]['temp'] = max_length

    # Variables to easily keep track of the length of chains in short_unique
    shortest = max_length
    next_shortest = max_length
    
    # Apply random stabilizers to start in high temperature state
    if randomize:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Generate chains
    for step in range(steps):
        # Do metropolis sampling
        chain.update_chain_fast(5)
        #print(len(short_unique[1]))
        # Convert the current qubit matrix to string for hashing
        key = hash(chain.code.qubit_matrix.tobytes())

        # Check if this error chain has already been seen by comparing hashes
        if key in unique_lengths:
            # Increment counter for chains of this length
            len_counts[unique_lengths[key]] += 1

        # If this chain is new, add it to dictionary of unique chains
        else:
            #print('\nfound new chain')
            #print('len before new len found:', len(short_unique[1]))
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
                #print('it has a new length!')
                # Initiate counter for chains of this length
                len_counts[unique_lengths[key]] = 1
                # Check if this chain is shorter than prevous shortest chain
                if length < shortest:
                    #print('Found new shortest chain')
                    # Then the previous shortest length is the new next shortest
                    next_shortest = shortest
                    shortest = length
                    
                    #print('before update:', len(short_unique[1]))
                    # Clear next shortest set and set i equal to shortest
                    short_unique[1].clear()
                    short_unique[1].update(short_unique[0])
                    #print('after update:', len(short_unique[1]))
                    # And the current length is the new shortest
                    short_unique[0].clear()
                    short_unique[0][key] = length
                
                # Otherwise, check if this chain is shorter than previous next shortest chain
                elif length < next_shortest:
                    #print('Found new next shortest chain!')
                    # Then reset stats of next shortest chain
                    next_shortest = length

                    # Clear and update next shortest set
                    short_unique[1].clear()
                    short_unique[1][key] = length
            #print('len(1) after new chain found:', len(short_unique[1]))

    return unique_lengths, len_counts, short_unique

 
def STRC(init_code, size, p_error, p_sampling=None, droplets=10, steps=20000):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # Create chains with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [Chain(size, p_sampling, copy.deepcopy(code)) for code in init_code]
        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chains with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain(size, p_sampling, copy.deepcopy(init_code))
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)
        # apply uniform stabilizers, i.e. rain
        randomize = True

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
        chain = eq_chains[eq]

        # Start parallel processes with droplets.
        if droplets == 1:
            unique_lengths, len_counts, short_unique = STRC_droplet((copy.deepcopy(chain), steps, max_length, eq, randomize))
            shortest = next(iter(short_unique[0].values()))
            next_shortest = next(iter(short_unique[1].values()))
        else:
            with Pool(droplets) as pool:
                output = pool.map(STRC_droplet, [(copy.deepcopy(chain), steps, max_length, eq, True) for _ in range(droplets-1)] + [(copy.deepcopy(chain), steps, max_length, eq, randomize)])

            # We need to combine the results from all raindrops
            unique_lengths = {}
            len_counts = {}
            short_unique = [{} for _ in range(2)]

            shortest = max_length
            next_shortest = max_length

            # Find shortest and next shortest length found by any chain
            for i in range(droplets):
                _,_,data = output[i]
                if next(iter(data[0].values())) < shortest:
                    next_shortest = shortest
                    shortest = next(iter(data[0].values()))
                if next(iter(data[1].values())) < next_shortest:
                    next_shortest = next(iter(data[1].values()))
            
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
                shortest_i = next(iter(short_unique_i[0].values()))
                next_shortest_i = next(iter(short_unique_i[1].values()))

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
    size = 9
    steps = size ** 4
    print(steps)
    #reader = MCMCDataReader('data/data_7x7_p_0.19.xz', size)
    p_error = 0.20
    p_sampling = 0.25
    init_code = Planar_code(size)
    tries = 2
    distrs = np.zeros((tries, init_code.nbr_eq_classes))
    mean_tvd = 0.0
    for i in range(10):
        init_code.generate_random_error(p_error)
        ground_state = init_code.define_equivalence_class()
        
        #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
        #init_qubit = np.copy(init_code.qubit_matrix)

        class_init = class_sorted_mwpm(init_code)
        init_qubit = [code.qubit_matrix for code in class_init]

        print('################ Chain', i+1 , '###################')
        
        for i in range(tries):
            #print('Try', i+1, ':', distrs[i], 'most_likeley_eq', np.argmax(distrs[i]), 'ground state:', ground_state)

            t0 = time.time()
            v1 = single_temp(copy.deepcopy(class_init), p=p_error, max_iters=steps)
            print('Try single_temp', i+1, ':', v1, 'most_likely_eq', np.argmin(v1), 'ground state:', ground_state, 'time taken: ', time.time()-t0)
            t0 = time.time()
            distrs[i] = STDC(copy.deepcopy(class_init), size=size, p_error=p_error, p_sampling=p_sampling, steps=steps, droplets=1)
            print('Try STDC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)
            t0 = time.time()
            distrs[i] = STRC(copy.deepcopy(class_init), size=size, p_error=p_error, p_sampling=p_sampling, steps=steps, droplets=1)
            print('Try STRC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)
            t0 = time.time()
            distrs[i] = PTEQ(copy.deepcopy(init_code), p=p_error)
            print('Try PTEQ       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'ground state:', ground_state, time.time()-t0)
            t0 = time.time()

        tvd = sum(abs(distrs[1]-distrs[0]))
        mean_tvd += tvd
        print('TVD:', tvd)
    print('Mean TVD:', mean_tvd / 10)

        #print('STRC distribution 1:', distr1)
        #print('STRC distribution 2:', distr2)
