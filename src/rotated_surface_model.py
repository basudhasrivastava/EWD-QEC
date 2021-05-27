import numpy as np
from random import random
from numba import njit
import random as rand


class RotSurCode():
    nbr_eq_classes = 4

    def __init__(self, size):
        self.system_size = size
        self.qubit_matrix = np.zeros((self.system_size, self.system_size), dtype=np.uint8)

    # def generate_random_error(self, p_error):
    #     qubits = np.random.uniform(0, 1, size=(self.system_size, self.system_size))
    #     no_error = qubits > p_error
    #     error = qubits < p_error
    #     qubits[no_error] = 0
    #     qubits[error] = 1
    #     pauli_error = np.random.randint(3, size=(self.system_size, self.system_size)) + 1
    #     self.qubit_matrix[:, :] = np.multiply(qubits, pauli_error)

    def generate_random_error(self, p_error, eta):  # Z-biased noise
        eta = eta
        p = p_error
        p_z = p * eta / (eta + 1)
        p_x = p / (2 * (eta + 1))
        p_y = p_x
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

    # def generate_random_error(self, p_error, eta):  # Y-biased noise
    #     eta = eta
    #     p = p_error
    #     p_y = p * eta / (eta + 1)
    #     p_x = p / (2 * (eta + 1))
    #     p_z = p_x
    #     size = self.system_size
    #     for i in range(size):
    #         for j in range(size):
    #             q = 0
    #             r = rand.random()
    #             if r < p_y:
    #                 q = 2
    #             elif p_y < r < (p_y + p_x):
    #                 q = 1
    #             elif (p_y + p_x) < r < (p_y + p_x + p_z):
    #                 q = 3
    #             self.qubit_matrix[i, j] = q

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


@njit('(uint8[:,:], int64, int64, int64)')  # Z-biased noise
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
            old_qubit = result_qubit_matrix[i, X_pos]
            new_qubit = 1 ^ old_qubit
            result_qubit_matrix[i, X_pos] = new_qubit
            if old_qubit and not new_qubit:
                error_count -= 1
            elif new_qubit and not old_qubit:
                error_count += 1
    if do_Z:
        for i in range(size):
            old_qubit = result_qubit_matrix[Z_pos, i]
            new_qubit = 3 ^ old_qubit
            result_qubit_matrix[Z_pos, i] = new_qubit
            if old_qubit and not new_qubit:
                error_count -= 1
            elif new_qubit and not old_qubit:
                error_count += 1

    return result_qubit_matrix, error_count


# @njit('(uint8[:,:], int64, int64, int64)')  # Y-biased noise
# def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
#     result_qubit_matrix = np.copy(qubit_matrix)
#     error_count = 0
#
#     if operator == 0:
#         return result_qubit_matrix, 0
#     size = qubit_matrix.shape[0]
#
#     do_X = (operator == 1)
#     do_Z = (operator == 3)
#     do_Y = (operator == 2)
#
#
#     if do_X:
#         for i in range(size):
#             old_qubit = result_qubit_matrix[i, X_pos]
#             new_qubit = 1 ^ old_qubit
#             result_qubit_matrix[i, X_pos] = new_qubit
#             if old_qubit and not new_qubit:
#                 error_count -= 1
#             elif new_qubit and not old_qubit:
#                 error_count += 1
#     if do_Z:
#         for i in range(size):
#             old_qubit = result_qubit_matrix[Z_pos, i]
#             new_qubit = 3 ^ old_qubit
#             result_qubit_matrix[Z_pos, i] = new_qubit
#             if old_qubit and not new_qubit:
#                 error_count -= 1
#             elif new_qubit and not old_qubit:
#                 error_count += 1
#     if do_Y:
#         for i in range(size):
#             for j in range(size):
#                 old_qubit = result_qubit_matrix[i, j]
#                 new_qubit = 2 ^ old_qubit
#                 result_qubit_matrix[i, j] = new_qubit
#                 if old_qubit and not new_qubit:
#                     error_count -= 1
#                 elif new_qubit and not old_qubit:
#                     error_count += 1
#
#     return result_qubit_matrix, error_count


@njit('(uint8[:,:],)')
def _apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[0]

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

    size = qubit_matrix.shape[0]
    result_qubit_matrix = np.copy(qubit_matrix)
    error_count = 0
    op = 0

    if operator == 1:  # full
        qarray = [[0 + row, 0 + col], [0 + row, 1 + col], [1 + row, 0 + col], [1 + row, 1 + col]]
        if row % 2 == 0:
            if col % 2 == 0:
                op = 1
            else:
                op = 3
        else:
            if col % 2 == 0:
                op = 3
            else:
                op = 1
    elif operator == 3:  # half
        if col == 0:
            op = 1
            qarray = [[0, row*2 + 1], [0, row*2 + 2]]
        elif col == 1:
            op = 3
            qarray = [[row*2 + 1, size - 1], [row*2 + 2, size - 1]]
        elif col == 2:
            op = 1
            qarray = [[size - 1, row*2], [size - 1, row*2 + 1]]
        elif col == 3:
            op = 3
            qarray = [[row*2, 0], [row*2 + 1, 0]]

    for i in qarray:
        old_qubit = result_qubit_matrix[i[0], i[1]]
        new_qubit = op ^ old_qubit
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

    x_errors = np.count_nonzero(qubit_matrix[0, :] == 1)
    x_errors += np.count_nonzero(qubit_matrix[0, :] == 2)

    z_errors = np.count_nonzero(qubit_matrix[:, 0] == 3)
    z_errors += np.count_nonzero(qubit_matrix[:, 0] == 2)

    return (x_errors % 2) + 2 * (z_errors % 2)
