import numpy as np
import random as rand
import copy

from numba import jit, njit
from .planar_model import _apply_random_stabilizer  # ???
import pandas as pd


class Chain_alpha:
    def __init__(self, pz_tilde, alpha, code):
        self.code = code
        self.pz_tilde = pz_tilde
        self.alpha = alpha
        self.p_logical = 0
        self.flag = 0

        xb = np.count_nonzero(self.code.qubit_matrix[:, :] == 1)
        yb = np.count_nonzero(self.code.qubit_matrix[:, :] == 2)
        zb = np.count_nonzero(self.code.qubit_matrix[:, :] == 3)

        self.n_eff = zb + self.alpha*(xb + yb)
        # self.factor = ((self.p / 3.0) / (1.0 - self.p))  # rename me TODO
    # runs iters number of steps of the metroplois-hastings algorithm

    def update_chain(self, iters):
        num = self.code.system_size**2
        alpha = self.alpha

        pz_tilde = self.pz_tilde
        p_tilde = pz_tilde + 2*pz_tilde**alpha
        p = p_tilde / (1+p_tilde)
        pz = pz_tilde*(1-p)
        px = py = pz_tilde**alpha * (1-p)

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
                    self.n_eff = zn + self.alpha*(xn + yn)
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
                    self.n_eff = zn + self.alpha*(xn + yn)


    def update_chain_fast(self, iters):
        self.code.qubit_matrix = _update_chain_fast(self.code.qubit_matrix, self.factor, iters)


class Ladder_alpha:
    def __init__(self, pz_tilde_bottom, init_code, alpha, Nc, p_logical=0):
        
        self.alpha = alpha
        
        self.pz_tilde_bottom = pz_tilde_bottom

        # seed code
        self.init_code = init_code

        # number of chains
        self.Nc = Nc

        # logical sampling rate in top chain
        self.p_logical = p_logical
        
        # pz_tilde_top is 0.75 for depolarizing noise
        pz_tilde_top = 1

        # temporary list of sampling probabilities
        pz_tilde_ladder = np.linspace(pz_tilde_bottom, pz_tilde_top, Nc)
        self.pz_tilde_ladder = pz_tilde_ladder

        # list of relative probabilities
        self.pz_tilde_diff = (pz_tilde_ladder[:-1] * (1 - pz_tilde_ladder[1:])) / (pz_tilde_ladder[1:] * (1 - pz_tilde_ladder[:-1]))
        # list of Chains of increasing p
        self.chains = [Chain_alpha(pz_tilde, alpha, copy.deepcopy(init_code)) for pz_tilde in pz_tilde_ladder]

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
        pz_tilde_low = self.chains[ind_lo].pz_tilde
        pz_tilde_high = self.chains[ind_lo+1].pz_tilde
        n_eff_low = self.chains[ind_lo].n_eff
        n_eff_high = self.chains[ind_lo+1].n_eff
        return rand.random() < (pz_tilde_low/pz_tilde_high)**(n_eff_high-n_eff_low)
        # relative probabilities between chains (except exponent)
        #print('relp',rel_p)
        #return _r_flip(ne_lo, ne_hi, rel_p)

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
    #print(rel_p ** (ne_hi - ne_lo))
    return rand.random() < rel_p ** (ne_hi - ne_lo)


@njit(cache=True)
def _update_chain_fast(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, qubit_errors_change = _apply_random_stabilizer(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** qubit_errors_change:
            qubit_matrix = new_matrix
    return qubit_matrix
