import numpy as np
import random as rand
import copy
import pandas as pd

from numba import njit

from src.xzzx_model import xzzx_code, _apply_random_stabilizer as apply_stabilizer_fast_xzzx
from src.rotated_surface_model import RotSurCode, _apply_random_stabilizer as apply_stabilizer_fast_rotated
from src.planar_model import Planar_code, _apply_random_stabilizer as apply_stabilizer_fast_planar
from src.toric_model import Toric_code, _apply_random_stabilizer as apply_stabilizer_fast_toric

class Chain_alpha:
    def __init__(self, code, pz_tilde, alpha):
        self.code = code
        self.pz_tilde = pz_tilde
        self.alpha = alpha
        self.p_logical = 0
        self.flag = 0
    
    # runs iters number of steps of the metroplois-hastings algorithm
    def update_chain(self, iters):

        if self.p_logical != 0:
            for _ in range(iters):
                # apply logical or stabilizer with p_logical
                if rand.random() < self.p_logical:
                    new_matrix, (dx, dy, dz) = self.code.apply_random_logical()
                else:
                    new_matrix, (dx, dy, dz) = self.code.apply_random_stabilizer()
                
                if rand.random() < self.pz_tilde**(dz + self.alpha*(dx + dy)):
                    self.code.qubit_matrix = new_matrix

        else:
            for _ in range(iters):
                new_matrix, (dx, dy, dz) = self.code.apply_random_stabilizer()

                if rand.random() < self.pz_tilde**(dz + self.alpha*(dx + dy)):
                    self.code.qubit_matrix = new_matrix

    def update_chain_fast(self, iters):
        if isinstance(self.code, xzzx_code):
            self.code.qubit_matrix = _update_chain_fast_xzzx(self.code.qubit_matrix, self.pz_tilde, self.alpha, iters)
        elif isinstance(self.code, RotSurCode):
            self.code.qubit_matrix = _update_chain_fast_rotated(self.code.qubit_matrix, self.pz_tilde, self.alpha, iters)
        elif isinstance(self.code, Planar_code):
            self.code.qubit_matrix = _update_chain_fast_planar(self.code.qubit_matrix, self.pz_tilde, self.alpha, iters)
        elif isinstance(self.code, Toric_code):
            self.code.qubit_matrix = _update_chain_fast_toric(self.code.qubit_matrix, self.pz_tilde, self.alpha, iters)
        else:
            raise ValueError("Fast chain updates not available for this code")

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
        self.chains = [Chain_alpha(copy.deepcopy(init_code) ,pz_tilde, alpha) for pz_tilde in pz_tilde_ladder]

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


@njit(cache=True)
def _update_chain_fast_xzzx(qubit_matrix, pz_tilde, alpha, iters):

    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_xzzx(qubit_matrix)
        
        p = pz_tilde**(dz + alpha*(dx + dy))
        if p > 1 or rand.random() < p:
            return new_matrix
        else:
            return qubit_matrix

@njit(cache=True)
def _update_chain_fast_rotated(qubit_matrix, pz_tilde, alpha, iters):

    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_rotated(qubit_matrix)

        p = pz_tilde**(dz + alpha*(dx + dy))
        if p > 1 or rand.random() < p:
            return new_matrix
        else:
            return qubit_matrix

@njit(cache=True)
def _update_chain_fast_planar(qubit_matrix, pz_tilde, alpha, iters):

    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_planar(qubit_matrix)

        p = pz_tilde**(dz + alpha*(dx + dy))
        if p > 1 or rand.random() < p:
            return new_matrix
        else:
            return qubit_matrix

@njit(cache=True)
def _update_chain_fast_toric(qubit_matrix, pz_tilde, alpha, iters):

    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_toric(qubit_matrix)
        
        p = pz_tilde**(dz + alpha*(dx + dy))
        if p > 1 or rand.random() < p:
            return new_matrix
        else:
            return qubit_matrix
