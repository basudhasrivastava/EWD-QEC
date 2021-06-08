import numpy as np
import random as rand
import copy
import collections
import time
import matplotlib.pyplot as plt

from numba import jit, njit
from src.toric_model import Toric_code
from src.planar_model import Planar_code
# from src.rotated_surface_model_dep import RotSurCodeDep
# from src.rotated_surface_model import RotSurCode
from src.xzzx_model import xzzx_code
# from src.xyzxyz_model import xyz_code
from src.mcmc_biased import *
from src.mcmc_alpha import *
from src.mwpm import class_sorted_mwpm, regular_mwpm
import pandas as pd
import time

from math import log, exp
from multiprocessing import Pool
from operator import itemgetter
# SEQ = 2, eps = 0.1
# TOPS = 10, tops_burn = 2


def PTEQ_biased(init_code, p, eta=0.5, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    Nc = Nc or init_code.system_size
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')
    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)
    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)
    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0
    # Convergence flag
    convergence_reached = False
    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder_biased(p, init_code, eta, Nc, 0.5)
    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)
        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()
        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in
            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder.chains[0].code.count_errors()
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1
        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT_biased(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: PTEQ hit max number of steps before convergence:\t', step + 1, '\n\n')
    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based_PT_biased(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
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


def PTEQ_alpha(init_code, pz_tilde, alpha=1, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    Nc = Nc or init_code.system_size
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')
    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)
    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)
    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0
    # Convergence flag
    convergence_reached = False
    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder_alpha(pz_tilde, init_code, alpha, Nc, 0.5)
    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)
        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()
        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in
            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder.chains[0].code.count_errors()
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1
        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT_alpha(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: PTEQ hit max number of steps before convergence:\t', step + 1, '\n\n')
    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based_PT_alpha(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
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


if __name__ == '__main__':
    size = 5
    steps = 10 * size ** 4  # 10 * size ** 4
    # p_sampling = 0.30
    init_code = xzzx_code(size)
    loss = np.empty(0)
    eta = 10


    pz_tilde = 0.1
    alpha = 2
    
    
    p_rates = range(5, 51, 5)
    for p in p_rates:
        p_error = p/100
        los = 0.
        des = 1
        for m in range(des):
            # distrs = np.zeros(init_code.nbr_eq_classes)
            init_code.generate_random_error(p_error, eta)
            # init_code.generate_known_error(p_error)
            ground_state = init_code.define_equivalence_class()
            # mwpm_init = regular_mwpm(init_code)
            
            
            #distrs = PTEQ_biased(copy.deepcopy(init_code), p=p_error, eta=eta)
            distrs = PTEQ_alpha(copy.deepcopy(init_code), pz_tilde=pz_tilde, alpha=alpha)
            
            # if mwpm_init != ground_state:
            #     los = los + 1
            if np.argmax(distrs) != ground_state:
                los = los + 1
            # if m != 0:
            #     print(p_error*100, m, los*100/m)
        print(p_error*100, los*100/des)
        loss = np.append(loss, los*100/des)
    print(loss)
