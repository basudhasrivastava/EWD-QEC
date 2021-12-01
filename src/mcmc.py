import numpy as np
import random as rand
import copy
import pandas as pd

from numba import njit

from src.xzzx_model import xzzx_code, _apply_random_stabilizer as apply_stabilizer_fast_xzzx
from src.rotated_surface_model import RotSurCode, _apply_random_stabilizer as apply_stabilizer_fast_rotated
from src.planar_model import Planar_code, _apply_random_stabilizer as apply_stabilizer_fast_planar, _count_errors_xyz
from src.toric_model import Toric_code, _apply_random_stabilizer as apply_stabilizer_fast_toric


class Chain:
    def __init__(self, p, code):
        self.code = code
        self.p = p
        self.p_logical = 0
        self.flag = 0
        self.factor = ((self.p / 3.0) / (1.0 - self.p))  # rename me 

    # runs iters number of steps of the metroplois-hastings algorithm
    def update_chain(self, iters):
        if self.p_logical != 0:
            for _ in range(iters):
                # apply logical or stabilizer with p_logical
                if rand.random() < self.p_logical:
                    new_matrix, (dx, dy, dz) = self.code.apply_random_logical()
                else:
                    new_matrix, (dx, dy, dz) = self.code.apply_random_stabilizer()

                qubit_errors_change = dx + dy + dz

                # Avoid calculating r if possible. If self.p is 0.75 r = 1 and we accept all changes
                # If the new qubit matrix has equal or fewer errors, r >= 1 and we also accept all changes
                if self.p >= 0.75 or qubit_errors_change <= 0:
                    self.code.qubit_matrix = new_matrix
                    continue
                # acceptence ratio
                if rand.random() < self.factor ** qubit_errors_change:
                    self.code.qubit_matrix = new_matrix

        else:
            for _ in range(iters):
                new_matrix, (dx, dy, dz) = self.code.apply_random_stabilizer()

                qubit_errors_change = dx + dy + dz

                # acceptence ratio
                if rand.random() < self.factor ** qubit_errors_change:
                    self.code.qubit_matrix = new_matrix

    def update_chain_fast(self, iters):
        if isinstance(self.code, xzzx_code):
            self.code.qubit_matrix = _update_chain_fast_xzzx(self.code.qubit_matrix, self.factor, iters)
        elif isinstance(self.code, RotSurCode):
            self.code.qubit_matrix = _update_chain_fast_rotated(self.code.qubit_matrix, self.factor, iters)
        elif isinstance(self.code, Planar_code):
            self.code.qubit_matrix = _update_chain_fast_planar(self.code.qubit_matrix, self.factor, iters)
        elif isinstance(self.code, Toric_code):
            self.code.qubit_matrix = _update_chain_fast_toric(self.code.qubit_matrix, self.factor, iters)
        else:
            raise ValueError("Fast chain updates not available for this code")


class Ladder:
    def __init__(self, p_bottom, init_code, Nc, p_logical=0):
        # sampling probability of bottom chain
        self.p_bottom = p_bottom

        # seed code
        self.init_code = init_code

        # number of chains
        self.Nc = Nc

        # logical sampling rate in top chain
        self.p_logical = p_logical
        p_top = 0.75

        # temporary list of sampling probabilities
        p_ladder = np.linspace(p_bottom, p_top, Nc)
        self.p_ladder = p_ladder

        # list of relative probabilities
        self.p_diff = (p_ladder[:-1] * (1 - p_ladder[1:])) / (p_ladder[1:] * (1 - p_ladder[:-1]))

        # list of Chains of increasing p
        self.chains = [Chain(p, copy.deepcopy(init_code)) for p in p_ladder]

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


class Chain_xyz:
    def __init__(self, p_xyz, code):
        self.code = code
        self.p_xyz = p_xyz
        self.factors = self.p_xyz / (1.0 - self.p_xyz.sum())  # rename me
        self.qubit_errors = code.count_errors_xyz()

    def update_chain_fast(self, iters):
        self.code.qubit_matrix, self.qubit_errors = _update_chain_fast_xyz(self.code.qubit_matrix, self.qubit_errors, self.factors, iters)


@njit('(int64, int64, float64)')
def _r_flip(ne_lo, ne_hi, rel_p):
    if ne_hi < ne_lo:
        return True
    else:
        return rand.random() < rel_p ** (ne_hi - ne_lo)


@njit(cache=True)
def _update_chain_fast_xzzx(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_xzzx(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** (dx + dy + dz):
            qubit_matrix = new_matrix
    return qubit_matrix

@njit(cache=True)
def _update_chain_fast_rotated(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_rotated(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** (dx + dy + dz):
            qubit_matrix = new_matrix
    return qubit_matrix

@njit(cache=True)
def _update_chain_fast_planar(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_planar(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** (dx + dy + dz):
            qubit_matrix = new_matrix
    return qubit_matrix

@njit(cache=True)
def _update_chain_fast_toric(qubit_matrix, factor, iters):
    for _ in range(iters):
        new_matrix, (dx, dy, dz) = apply_stabilizer_fast_toric(qubit_matrix)

        # acceptence ratio
        if rand.random() < factor ** (dx + dy + dz):
            qubit_matrix = new_matrix
    return qubit_matrix


@njit(cache=True)
def _update_chain_fast_xyz(qubit_matrix, qubit_errors, factors, iters):
    for _ in range(iters):
        new_matrix, _ = _apply_random_stabilizer(qubit_matrix)
        qubit_errors_new = _count_errors_xyz(new_matrix)
        qubit_errors_change = qubit_errors_new - qubit_errors

        # acceptence ratio
        if rand.random() < (factors ** qubit_errors_change).prod():
            qubit_matrix = new_matrix
            qubit_errors = qubit_errors_new
    return qubit_matrix, qubit_errors
