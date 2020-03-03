from src.toric_model import Toric_code
from src.mcmc import *
import numpy as np
import copy

def main():
    size = 5
    toric_init = Toric_code(size)

    '''
    # Initial error configuration
    init_errors = np.array([[1, 0, 1, 3], [1, 1, 0, 1], [1, 3, 3, 3], [1, 4, 1, 1], [1, 1, 1, 1]])
    for arr in init_errors:  # apply the initial error configuration
        print(arr)
        action = Action(position=arr[:3], action=arr[3])
        toric_init.step(action)
    '''
    # create the diffrent chains in an array
    N = 15  # number of chains in ladder, must be odd
    try:
        N % 2 == 0
    except:
        print('Number of chains was not odd.')
    ladder = []  # ladder to store all chains
    p_start = 0.1  # what should this really be???
    p_end = 0.75  # p at top chain as per high-threshold paper
    tops0 = 0
    SEQ = 2
    TOPS = 5
    tol = 0.1
    convergence_reached = 0
    nbr_errors_bottom_chain = []
    ec_frequency = []
    min_quarter_width = 0

    # test random error initialisation
    toric_init.generate_random_error(p_start)
    toric_init.qubit_matrix = apply_stabilizers_uniform(toric_init.qubit_matrix)

    # plot initial error configuration
    toric_init.plot_toric_code(toric_init.next_state, 'Chain_init')

    # add and copy state for all chains in ladder
    for i in range(N):
        p_i = p_start + ((p_end - p_start) / (N - 1)) * i
        ladder.append(Chain(size, p_i))
        ladder[i].toric = copy.deepcopy(toric_init)  # give all the same initial state
    ladder[N - 1].p_logical = 0.5  # set top chain as the only one where logicals happen

    steps = input('How many steps? Ans: ')
    iters = input('How many iterations for each step? Ans: ')
    while not (steps == 'n' or iters == 'n'):
        try:
            steps = int(steps)
            iters = int(iters)
        except:
            print('Input data bad, using default of 1 for both.')
            steps = 1
            iters = 1

        bottom_equivalence_classes = np.zeros(steps, dtype=int)

        for j in range(steps):
            # run mcmc for each chain [steps] times
            for i in range(N):
                for _ in range(iters):
                    ladder[i].update_chain()
            # now attempt flips from the top down
            for i in reversed(range(N - 1)):
                if i == (N - 2):
                    ladder[i + 1].flag = 1
                if ladder[0].flag == 1:
                    tops0 += 1
                    ladder[0].flag = 0
                r_flip(ladder[i], ladder[i + 1])
            if tops0>= TOPS:
                temp = np.count_nonzero(ladder[0].toric.qubit_matrix)
                nbr_errors_bottom_chain.append(temp)  # vill man räkna y som två fel?
                min_quarter_width += 1
                if min_quarter_width > 10:
                    second_quarter = nbr_errors_bottom_chain[(len(nbr_errors_bottom_chain) // 4): (len(nbr_errors_bottom_chain) // 4) * 2]
                    fourth_quarter = nbr_errors_bottom_chain[(len(nbr_errors_bottom_chain) // 4) * 3: (len(nbr_errors_bottom_chain) // 4) * 4]
                    Average_second_quarter = sum(second_quarter) / (len(second_quarter))
                    Average_fourth_quarter = sum(fourth_quarter) / (len(fourth_quarter))
                    error = abs(Average_second_quarter - Average_fourth_quarter)
                    if convergence_reached == 1:
                        ec_frequency.append(define_equivalence_class(ladder[0].toric.qubit_matrix))
                    if error > tol:
                        tops0== TOPS
                        ec_frequency.append(define_equivalence_class(ladder[0].toric.qubit_matrix))
                    if tops0== tops0+ SEQ:
                        if convergence_reached == 0:
                            print('Convergence achieved.')
                        convergence_reached = 1
            # record current equivalence class in bottom layer

            bottom_equivalence_classes[j] = define_equivalence_class(ladder[0].toric.qubit_matrix)

        # plot all chains
        for i in range(N):
            ladder[i].plot('Chain_' + str(i))

        # count number of occurrences of each equivalence class
        # equivalence_class_count[i] is the number of occurences of equivalence class number 'i'
        # if
        equivalence_class_count = np.bincount(bottom_equivalence_classes, minlength=15)

        print('Equivalence classes: \n', np.arange(16))
        print('Count:\n', equivalence_class_count)

        steps = input('How many steps? Ans: ')
        iters = input('How many iterations for each step? Ans: ')


if __name__ == "__main__":
    main()
