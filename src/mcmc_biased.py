import numpy as np
import random as rand
import copy

from numba import jit, njit
from .planar_model import _apply_random_stabilizer  # ???
import pandas as pd


class Chain_biased:
    def __init__(self, p, eta, code):
        self.code = code
        self.p = p
        self.eta = eta
        self.p_logical = 0
        self.flag = 0
        self.factor = ((self.p / 3.0) / (1.0 - self.p))  # rename me
    # runs iters number of steps of the metroplois-hastings algorithm

    def update_chain(self, iters):
        num = self.code.system_size**2
        eta = self.eta
        p_top = (eta + 1) / (2 * eta + 1)
        p = self.p
        pz = p * eta / (eta + 1)
        px = p / (2 * (eta + 1))
        py = px
        xb = np.count_nonzero(self.code.qubit_matrix[:, :] == 1)
        yb = np.count_nonzero(self.code.qubit_matrix[:, :] == 2)
        zb = np.count_nonzero(self.code.qubit_matrix[:, :] == 3)
        pb = (px ** xb) * (py ** yb) * (pz ** zb) * ((1 - px - py - pz) ** (num - xb - yb - zb))
        if self.p_logical != 0:
            for _ in range(iters):
                # apply logical or stabilizer with p_logical
                if rand.random() < self.p_logical:
                    new_matrix, qubit_errors_change = self.code.apply_random_logical()
                else:
                    new_matrix, qubit_errors_change = self.code.apply_random_stabilizer()
                
                xn = np.count_nonzero(new_matrix[:, :] == 1)
                yn = np.count_nonzero(new_matrix[:, :] == 2)
                zn = np.count_nonzero(new_matrix[:, :] == 3)
                pn = (px ** xn) * (py ** yn) * (pz ** zn) * ((1 - px - py - pz) ** (num - xn - yn - zn))
                ratio_p = pn / pb
                if rand.random() < ratio_p:
                    self.code.qubit_matrix = new_matrix
                    continue

        else:
            for _ in range(iters):
                new_matrix, qubit_errors_change = self.code.apply_random_stabilizer()
                
                xn = np.count_nonzero(new_matrix[:, :] == 1)
                yn = np.count_nonzero(new_matrix[:, :] == 2)
                zn = np.count_nonzero(new_matrix[:, :] == 3)
                pn = (px ** xn) * (py ** yn) * (pz ** zn) * ((1 - px - py - pz) ** (num - xn - yn - zn))
                ratio_p = pn / pb
                if rand.random() < ratio_p:
                    self.code.qubit_matrix = new_matrix


    def update_chain_fast(self, iters):
        self.code.qubit_matrix = _update_chain_fast(self.code.qubit_matrix, self.factor, iters)


class Ladder_biased:
    def __init__(self, p_bottom, init_code, eta, Nc, p_logical=0):
        
        self.eta = eta
        
        self._bottom = p_bottom

        # seed code
        self.init_code = init_code

        # number of chains
        self.Nc = Nc

        # logical sampling rate in top chain
        self.p_logical = p_logical
        
        # p_top is 0.75 for depolarizing noise
        p_top = (eta + 1) / (2 * eta + 1)

        # temporary list of sampling probabilities
        p_ladder = np.linspace(p_bottom, p_top, Nc)
        self.p_ladder = p_ladder

        # list of relative probabilities
        self.p_diff = (p_ladder[:-1] * (1 - p_ladder[1:])) / (p_ladder[1:] * (1 - p_ladder[:-1]))

        # list of Chains of increasing p
        self.chains = [Chain_biased(p, eta, copy.deepcopy(init_code)) for p in p_ladder]

        # special properties of top chain
        self.chains[-1].flag = 1
        self.chains[-1].p_logical = p_logical

        # count of chains that have "fallen all the way down"
        self.tops0 = 0

    def update_ladder(self, iters):
        for chain in self.chains:
            chain.update_chain(iters)

    # returns true if flip should be performed
    def r_flip(self, ind_lo):
        # chain lengths
        ne_lo = self.chains[ind_lo].code.count_errors()
        ne_hi = self.chains[ind_lo + 1].code.count_errors()
        # relative probabilities between chains (except exponent)
        rel_p = self.p_diff[ind_lo]
        return _r_flip(ne_lo, ne_hi, rel_p)

    def step(self, iters):
        self.update_ladder(iters)
        for i in reversed(range(self.Nc - 1)):
            if self.r_flip(i):
                self.chains[i].code, self.chains[i + 1].code = self.chains[i + 1].code, self.chains[i].code
                self.chains[i].flag, self.chains[i + 1].flag = self.chains[i + 1].flag, self.chains[i].flag
        self.chains[-1].flag = 1
        if self.chains[0].flag == 1:
            self.tops0 += 1
            self.chains[0].flag = 0


# This is the object we crate to read a file during training
class MCMCDataReader:
    def __init__(self, file_path, size):
        # file_path needs to be dataframe in pickle format
        self.__file_path = file_path
        # size is the size of the toric code
        self.__size = size
        try:
            self.__df = pd.read_pickle(file_path)
            self.__capacity = self.__df.index[-1][0] + 1  # The number of data samples in the dataset
        except:  # TODO fix exception
            print('No input file for MCMCDataReader')
        self.__current_index = 0

    def full(self):
        return self.__df.to_numpy().ravel()

    def has_next(self):
        return self.__current_index < self.__capacity

    def current_index(self):
        return self.__current_index

    def get_capacity(self):
        return self.__capacity


@njit('(int64, int64, float64)')
def _r_flip(ne_lo, ne_hi, rel_p):
    return rand.random() < rel_p ** (ne_hi - ne_lo)


@njit(cache=True)
def _update_chain_fast(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, qubit_errors_change = _apply_random_stabilizer(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** qubit_errors_change:
            qubit_matrix = new_matrix
    return qubit_matrix
