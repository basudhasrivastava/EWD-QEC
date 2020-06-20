import numpy as np
import copy

from src.toric_model import Toric_code
from src.mcmc import Chain, define_equivalence_class, apply_logical

from math import log, exp
from tqdm import tqdm

# add eq-crit that runs until a certain number of classes are found or not?
# separate eq-classes? qubitlist for diffrent eqs
# vill göra detta men med mwpm? verkar finnas sätt att hitta "alla" kortaste, frågan är om man även kan hitta alla längre också
# https://stackoverflow.com/questions/58605904/finding-all-paths-in-weighted-graph-from-node-a-to-b-with-weight-of-k-or-lower
# i nuläget kommer "bra eq" att bli straffade eftersom att de inte kommer få chans att generera lika många unika kedjor --bör man sätta något tak? eller bara ta med de kortaste inom varje?


def single_temp_direct_sum(qubit_matrix, size, p, steps=20000):
    init_toric = Toric_code(size)
    init_toric.qubit_matrix = qubit_matrix
    ladder = []  # list of chain objects

    qubitlist = []

    for i in range(16):
        ladder.append(Chain(init_toric.system_size, p))  # this p needs not be the same as p, as it is used to determine how we sample N(n)
        ladder[i].toric = copy.deepcopy(init_toric)  # give all the same initial state
        ladder[i].toric.qubit_matrix = apply_logical_operator(ladder[i].toric.qubit_matrix, i)  # apply different logical operator to each chain
        # here we start in a high entropy state for most eqs, which is not desired as it increases time to find smaller solutions.
    for i in tqdm(range(16)):
        for _ in range(int(steps*0.1)):
            ladder[i].update_chain(5)
    for i in tqdm(range(16)):
        for _ in range(int(steps*0.9)):
            ladder[i].update_chain(5)
            qubitlist.append(ladder[i].toric.qubit_matrix)

    # Only consider unique elements
    print('Finding unique elements...')
    qubitlist = np.unique(qubitlist, axis=0)

    # --------Determine EC-Distrubution--------
    eqdistr = np.zeros(16)
    beta = -log((p / 3) / (1-p))

    for i in range(len(qubitlist)):
        eq = define_equivalence_class(qubitlist[i])
        eqdistr[eq] += exp(-beta*np.count_nonzero(qubitlist[i]))

    return [int(x * 100 / sum(eqdistr)) for x in eqdistr]


def apply_logical_operator(qubit_matrix, number):
    binary = "{0:4b}".format(number)
    for i in range(16):

        if binary[0] == '1':
            qubit_matrix, _ = apply_logical(qubit_matrix, operator=1, layer=0, X_pos=0, Z_pos=0)
        if binary[1] == '1':
            qubit_matrix, _ = apply_logical(qubit_matrix, operator=3, layer=0, X_pos=0, Z_pos=0)
        if binary[2] == '1':
            qubit_matrix, _ = apply_logical(qubit_matrix, operator=1, layer=1, X_pos=0, Z_pos=0)
        if binary[3] == '1':
            qubit_matrix, _ = apply_logical(qubit_matrix, operator=3, layer=1, X_pos=0, Z_pos=0)

        return qubit_matrix


if __name__ == '__main__':
    init_toric = Toric_code(5)
    p_error = 0.15
    init_toric.generate_random_error(p_error)
    print(single_temp_direct_sum(init_toric.qubit_matrix, size=5, p=p_error, steps=20000))
    print(init_toric.qubit_matrix)
