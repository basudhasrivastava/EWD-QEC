import numpy as np
import random as rand

from .toric_model import Toric_code
from .util import Action

rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)   # Identity = 0
                                                                                # pauli_x = 1
                                                                                # pauli_y = 2
                                                                                # pauli_z = 3

'''
def apply_stabilizer(toric_model, row=int, col=int, operator=int):
    # operator is 1 (X <-> vertex) or 3 (Z <-> plaquette)
    d = toric_model.system_size #input np.array of form (qubit_matrix=int, row=int, col=int, add_operator=int)	
    
    if operator == 1:
        action1 = Action(position = np.array([1, row, col]), action = operator)
        action2 = Action(position = np.array([1, row, (col-1)%d]), action = operator)
        action3 = Action(position = np.array([0, row, col]), action = operator)
        action4 = Action(position = np.array([0, (row-1)%d, col]), action = operator)
    elif operator == 3:
        action1 = Action(position = np.array([1, row, col]), action = operator)
        action2 = Action(position = np.array([0, row, col]), action = operator)
        action3 = Action(position = np.array([0, row, (col+1)%d]), action = operator)
        action4 = Action(position = np.array([1, (row+1)%(d), col]), action = operator)

    toric_model.step(action1)
    toric_model.step(action2)
    toric_model.step(action3)
    toric_model.step(action4)
'''
def test_apply_random_logical(qubit_matrix, size=int):
    operator = np.random.randint(1,4)
    if operator == 2:
        layer = np.random.randint(2)
        temp_qubit_matrix = test_apply_logical(qubit_matrix, size, layer, 1)
        return test_apply_logical(temp_qubit_matrix, size, layer, 3)
    else:
        return test_apply_logical(qubit_matrix,size,np.random.randint(2), operator)
    

def test_apply_logical(qubit_matrix, size=int, layer=int , operator=int):
    #  Applies a specific logical to the qubit matrix
    index = np.random.randint(size)
    qubit_matrix_layers = np.full(size, layer)
    if operator == 1:
        if layer == 0:
            cols = np.arange(size)
            rows = np.full(size, index)
        if layer == 1:
            rows = np.arange(size)
            cols = np.full(size, index)
    elif operator == 3:
        if layer == 1:
            cols = np.arange(size)
            rows = np.full(size, index)
        if layer == 0:
            rows = np.arange(size)
            cols = np.full(size, index)

        # the operator matters as well.... we con only do X OR Z on each layer once at at time
    
    
    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator,old_operators]

    result_qubit_matrix = np.copy(qubit_matrix)
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix

def test_apply_stabilizer(qubit_matrix, size=int, row=int, col=int, operator=int):
    # gives the resulting qubit error matrix from applying (row, col, operator) stabilizer
    # doesn't update input qubit_matrix
    if operator == 1:
        qubit_matrix_layers = np.array([1, 1, 0, 0])
        rows = np.array([row, row, row, (row-1)%size])
        cols = np.array([col, (col-1)%size, col, col])

    elif operator == 3:
        qubit_matrix_layers = np.array([1, 0, 0, 1])
        rows = np.array([row, row, row, (row+1)%size])
        cols = np.array([col, col, (col+1)%size, col])

    old_operators = qubit_matrix[qubit_matrix_layers, rows, cols]
    new_operators = rule_table[operator][old_operators]

    result_qubit_matrix = np.copy(qubit_matrix)
    result_qubit_matrix[qubit_matrix_layers, rows, cols] = new_operators

    return result_qubit_matrix


def test_apply_random_stabilizer(qubit_matrix, size):
    #select random coordinates where to apply operator
    row = np.random.randint(0, size) # gives int in [0, d-1]
    col = np.random.randint(0, size)
    operator = np.random.randint(0, 2) #we only care about X and Z, and Y is represented by 2. Therefore: 
    if operator == 0:
        operator = 3
    return test_apply_stabilizer(qubit_matrix, size, row, col, operator)

    
def apply_random_stabilizer(toric):
    toric.qubit_matrix = test_apply_random_stabilizer(toric.qubit_matrix, toric.system_size)


def apply_n_independent_random_stabilizers(toric, n=int):
    for i in range(0,n):
        apply_random_stabilizer(toric)


def apply_n_distinct_random_stabilizers(toric, n=int):
    stabilizer_array = np.zeros(2*toric.system_size**2)
    stabilizer_array[:n] = np.ones(n, dtype=bool)
    np.random.shuffle(stabilizer_array)

    stabilizer_matrix = stabilizer_array.reshape(toric.system_size, toric.system_size, 2)
    for index, stab in np.ndenumerate(stabilizer_matrix):
        if stab:
            operator = index[2]
            if operator == 0: 
                operator = 3
            toric.qubit_matrix = test_apply_stabilizer(toric.qubit_matrix, toric.system_size, index[0], index[1], operator)

def error_ratio(qubit_matrix_current, qubit_matrix_next, p=float):
    qubit_errors_current = np.count_nonzero(qubit_matrix_current)
    qubit_errors_new = np.count_nonzero(qubit_matrix_next)

    ratio = ((p / 3.0) / (1.0 - p)) ** (qubit_errors_new - qubit_errors_current)
    
    return ratio


def permute_error(qubit_matrix, size, p, p_logical):
    
    if np.random.rand() < p_logical:
        new_matrix = test_apply_random_logical(qubit_matrix, size)
    else:
        new_matrix = test_apply_random_stabilizer(qubit_matrix, size)

    r = error_ratio(qubit_matrix, new_matrix, p)
    if np.random.rand() < r:
        return new_matrix
    else:
        return qubit_matrix

def init_error(toric, qubit_matrix):
    toric.qubit_matrix = np.copy(qubit_matrix)
    toric.syndrom('next_state')

