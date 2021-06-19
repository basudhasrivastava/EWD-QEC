import numpy as np
import matplotlib.pyplot as plt
from random import random
from numba import njit
import random as rand


class xzzx_code():
    nbr_eq_classes = 4

    def __init__(self, size):
        self.system_size = size
        self.qubit_matrix = np.zeros((self.system_size, self.system_size), dtype=np.uint8)

    def generate_random_error(self, p_x, p_y, p_z):
        size = self.system_size
        for i in range(size):
            for j in range(size):
                q = 0
                r = rand.random()
                if r < p_z:
                    q = 3
                elif p_z < r < (p_z + p_x):
                    q = 1
                elif (p_z + p_x) < r < (p_z + p_x + p_y):
                    q = 2
                self.qubit_matrix[i, j] = q


    def generate_known_error(self, p_error):
        self.qubit_matrix[0, 1] = 1
        self.qubit_matrix[1, 1] = 1

    def count_errors(self):
        return _count_errors(self.qubit_matrix)

    def apply_logical(self, operator: int, X_pos=0, Z_pos=0):
        return _apply_logical(self.qubit_matrix, operator, X_pos, Z_pos)

    def apply_stabilizer(self, row: int, col: int, operator: int):
        return _apply_stabilizer(self.qubit_matrix, row, col, operator)

    def apply_random_logical(self):
        return _apply_random_logical(self.qubit_matrix)

    def apply_random_stabilizer(self):
        return _apply_random_stabilizer(self.qubit_matrix)

    def define_equivalence_class(self):
        return _define_equivalence_class(self.qubit_matrix)


@njit('(uint8[:,:],)')
def _count_errors(qubit_matrix):
    return np.count_nonzero(qubit_matrix)


# @njit('(uint8[:,:], int64, int64, int64)')
# def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
#     result_qubit_matrix = np.copy(qubit_matrix)
#     error_count = 0
#
#     if operator == 0:
#         return result_qubit_matrix, 0
#     size = qubit_matrix.shape[0]
#
#     do_X = (operator == 1 or operator == 2)
#     do_Z = (operator == 3 or operator == 2)
#
#     if do_X:
#         for i in range(size):
#             old_qubit = result_qubit_matrix[i, X_pos]
#             if X_pos % 2 == 0:
#                 if i % 2 == 0:
#                     op = 1
#                 else:
#                     op = 3
#             else:
#                 if i % 2 == 0:
#                     op = 3
#                 else:
#                     op = 1
#             new_qubit = op ^ old_qubit
#             result_qubit_matrix[i, X_pos] = new_qubit
#             if old_qubit and not new_qubit:
#                 error_count -= 1
#             elif new_qubit and not old_qubit:
#                 error_count += 1
#     if do_Z:
#         for i in range(size):
#             old_qubit = result_qubit_matrix[Z_pos, i]
#             if Z_pos % 2 == 0:
#                 if i % 2 == 0:
#                     op = 3
#                 else:
#                     op = 1
#             else:
#                 if i % 2 == 0:
#                     op = 1
#                 else:
#                     op = 3
#             new_qubit = op ^ old_qubit
#             result_qubit_matrix[Z_pos, i] = new_qubit
#             if old_qubit and not new_qubit:
#                 error_count -= 1
#             elif new_qubit and not old_qubit:
#                 error_count += 1
#
#     return result_qubit_matrix, error_count

@njit('(uint8[:,:], int64, int64, int64)')
def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
    result_qubit_matrix = np.copy(qubit_matrix)
    error_count = 0

    if operator == 0:
        return result_qubit_matrix, 0
    size = qubit_matrix.shape[0]

    do_X = (operator == 1 or operator == 2)
    do_Z = (operator == 3 or operator == 2)

    if do_X:
        for i in range(size):
            nr = size - 1
            old_qubit = result_qubit_matrix[i, nr - i]
            op = 1
            new_qubit = op ^ old_qubit
            result_qubit_matrix[i, nr - i] = new_qubit
            if old_qubit and not new_qubit:
                error_count -= 1
            elif new_qubit and not old_qubit:
                error_count += 1
    if do_Z:
        for i in range(size):
            old_qubit = result_qubit_matrix[i, i]
            op = 3
            new_qubit = op ^ old_qubit
            result_qubit_matrix[i, i] = new_qubit
            if old_qubit and not new_qubit:
                error_count -= 1
            elif new_qubit and not old_qubit:
                error_count += 1

    return result_qubit_matrix, error_count


# @njit('(uint8[:,:], int64, int64, int64)')
# def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
#     result_qubit_matrix = np.copy(qubit_matrix)
#     error_count = 0
#     if operator == 0:
#         return result_qubit_matrix, 0
#     size = qubit_matrix.shape[0]
#     do_X = (operator == 1 or operator == 2)
#     do_Z = (operator == 3 or operator == 2)
#
#     if do_Z:
#         for i in range(size):
#             old_qubit = result_qubit_matrix[i, i]
#             op = 3
#             new_qubit = op ^ old_qubit
#             result_qubit_matrix[i, i] = new_qubit
#             if old_qubit and not new_qubit:
#                 error_count -= 1
#             elif new_qubit and not old_qubit:
#                 error_count += 1
#
#     return result_qubit_matrix, error_count


@njit('(uint8[:,:],)')
def _apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[0]

    # operator to use, 2 (Y) will make both X and Z on the same layer. 0 is identity
    # one operator for each layer
    op = int(random() * 4)

    if op == 1 or op == 2:
        X_pos = int(random() * size)
    else:
        X_pos = 0
    if op == 3 or op == 2:
        Z_pos = int(random() * size)
    else:
        Z_pos = 0

    return _apply_logical(qubit_matrix, op, X_pos, Z_pos)


@njit('(uint8[:,:], int64, int64, int64)')
def _apply_stabilizer(qubit_matrix, row: int, col: int, operator: int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix

    size = qubit_matrix.shape[0]
    result_qubit_matrix = np.copy(qubit_matrix)
    error_count = 0

    if operator == 1:
        qarray = [[0 + row, 0 + col], [1 + row, 0 + col], [0 + row, 1 + col], [1 + row, 1 + col]]
        opr = [1, 3, 3, 1]
        j = 0
        for i in qarray:
            old_qubit = result_qubit_matrix[i[0], i[1]]
            new_qubit = opr[j] ^ old_qubit
            j += 1
            result_qubit_matrix[i[0], i[1]] = new_qubit
            if old_qubit and not new_qubit:
                error_count -= 1
            elif new_qubit and not old_qubit:
                error_count += 1
    if operator == 3:
        if col == 0:
            qarray = [[0, row*2 + 1], [0, row*2 + 2]]
            opr = [3, 1]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                if old_qubit and not new_qubit:
                    error_count -= 1
                elif new_qubit and not old_qubit:
                    error_count += 1
        elif col == 1:
            qarray = [[row*2 + 1, size - 1], [row*2 + 2, size - 1]]
            opr = [1, 3]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                if old_qubit and not new_qubit:
                    error_count -= 1
                elif new_qubit and not old_qubit:
                    error_count += 1
        elif col == 2:
            qarray = [[size - 1, row*2], [size - 1, row*2 + 1]]
            opr = [1, 3]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                if old_qubit and not new_qubit:
                    error_count -= 1
                elif new_qubit and not old_qubit:
                    error_count += 1
        elif col == 3:
            qarray = [[row*2, 0], [row*2 + 1, 0]]
            opr = [3, 1]
            j = 0
            for i in qarray:
                old_qubit = result_qubit_matrix[i[0], i[1]]
                new_qubit = opr[j] ^ old_qubit
                j += 1
                result_qubit_matrix[i[0], i[1]] = new_qubit
                if old_qubit and not new_qubit:
                    error_count -= 1
                elif new_qubit and not old_qubit:
                    error_count += 1

    return result_qubit_matrix, error_count


@njit('(uint8[:,:],)')
def _apply_random_stabilizer(qubit_matrix):
    size = qubit_matrix.shape[0]
    rows = int((size-1)*random())
    cols = int((size-1)*random())
    rows2 = int(((size - 1)/2) * random())
    cols2 = int(4 * random())
    phalf = (size**2 - (size-1)**2 - 1)/(size**2-1)
    if rand.random() > phalf:
        # operator = 1 = full stabilizer
        return _apply_stabilizer(qubit_matrix, rows, cols, 1)
    else:
        # operator = 3 = half stabilizer
        return _apply_stabilizer(qubit_matrix, rows2, cols2, 3)


@njit('(uint8[:,:],)')
def _define_equivalence_class(qubit_matrix):
    # # of x errors in the first column
    size = qubit_matrix.shape[0]
    x_errors = np.count_nonzero(qubit_matrix[0, :] == 2)
    z_errors = np.count_nonzero(qubit_matrix[:, 0] == 2)
    for i in range(size):
        if i % 2 == 0:
            if qubit_matrix[0, i] == 1:
                x_errors += 1
        else:
            if qubit_matrix[0, i] == 3:
                x_errors += 1

    # of z errors in the first row
    for i in range(size):
        if i % 2 == 0:
            if qubit_matrix[i, 0] == 3:
                z_errors += 1
        else:
            if qubit_matrix[i, 0] == 1:
                z_errors += 1

    # return the parity of the calculated #'s of errors
    return (x_errors % 2) + 2 * (z_errors % 2)
