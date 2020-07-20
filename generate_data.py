import copy  # not used
import os
import sys
import time

import numpy as np
import pandas as pd

from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.mcmc import *
from decoders import *
from src.mwpm import *


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, max_capacity=10**4, nbr_datapoints=10**6):

    if params['code'] == 'planar':
        nbr_eq_class = 4
    elif params['code'] == 'toric':
        nbr_eq_class = 16
    
    if params['method'] == "all":
        nbr_eq_class *= 3
    # Creates data file if there is none otherwise adds to it
    try:
        df = pd.read_pickle(file_path)
        nbr_existing_data = df.index[-1][0] + 1
    except:
        df = pd.DataFrame()
        nbr_existing_data = 0

    print('\nDataFrame with ' + str(nbr_existing_data) +
          ' datapoints opened at: ' + str(file_path))

    # Stop the file from exceeding the max limit of nbr of datapoints
    nbr_to_generate = min(max_capacity-nbr_existing_data, nbr_datapoints)
    if nbr_to_generate < nbr_datapoints:
        print('Generating ' + str(max(nbr_to_generate, 0))
              + ' datapoins instead of ' + str(nbr_datapoints)
              + ', as the given number would overflow existing file')

    df_list = []  # Initiate temporary list

    # Loop to generate data points
    for i in np.arange(nbr_to_generate) + nbr_existing_data:
        print('Starting generation of point nr: ' + str(i + 1))

        # Initiate code
        if params['code'] == 'toric':
            init_code = Toric_code(params['size'])
            init_code.generate_random_error(params['p_error'])
        elif params['code'] == 'planar':
            init_code = Planar_code(params['size'])
            init_code.generate_random_error(params['p_error'])
 
        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix.reshape((-1)))


        
        if params['mwpm_init']: #get mwpm starting points
            init_code = class_sorted_mwpm(init_code)
            print('Starting in MWPM state')
        else: #randomize input matrix, no trace of seed.
            init_code.qubit_matrix, _ = init_code.apply_random_logical()
            print('Starting in random state')

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if params['method'] == "PTEQ":
            df_eq_distr = PTEQ(init_code, params['p_error'])
        elif params['method'] == "STDC":
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr = STDC(init_code, params['size'], params['p_error'], params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
        elif params['method'] == "ST":
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr = single_temp(init_code, params['p_error'],params['steps'])
            df_eq_distr = np.array(df_eq_distr)
        elif params['method'] == "STRC":
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr = STRC(init_code, params['size'], params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
        elif params['method'] == "all":
            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr1 = single_temp(init_code, params['p_error'],params['steps'])

            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr2 = STDC(init_code, params['size'], params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])

            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            df_eq_distr3 = STRC(init_code, params['size'], params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])

            df_eq_distr = np.concatenate((df_eq_distr1,df_eq_distr2,df_eq_distr3), axis=0)
        elif params['method'] == "eMWPM":
            out = class_sorted_mwpm(copy.deepcopy(init_code))
            lens = np.zeros((4))
            for j in range(4):
                lens[j] = out[j].count_errors()
            choice = np.argmin(lens)
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            print(df_eq_distr)

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        

        # Create indices for generated data
        names = ['data_nr', 'layer', 'x', 'y']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(2),
                                                 np.arange(params['size']),
                                                 np.arange(params['size'])],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(nbr_eq_class)+2, [0],
                                                 [0]], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame(df_qubit.astype(np.uint8), index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame(df_eq_distr,
                                index=index_distr, columns=['data'])

        # Add dataframes to temporary list to shorten computation time
        
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list
        
        if (i + 1) % 100 == 0:
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)
    
    print('\nCompleted')


if __name__ == '__main__':

    # Get job array id, set working directory, set timer
    try:
        array_id = str(sys.argv[1])
        local_dir = str(sys.argv[2])
    except:
        array_id = '0'
        local_dir = '.'
        print('invalid sysargs')

    params = {'code': "planar",
            'method': "eMWPM",
            'size': 5,
            'p_error': np.round((0.05 + float(array_id) / 50), decimals=2),
            'p_sampling': 0.25,#np.round((0.05 + float(array_id) / 50), decimals=2),
            'droplets':4,
            'mwpm_init':False,
            'Nc':None,
            'iters': 10,
            'conv_criteria': 'error_based',
            'SEQ': 2,
            'TOPS': 10,
            'eps': 0.1}
    params.update({'steps': int((10000 * (params['size'] / 5) ** 4))})

    print(params['steps'])

    # Build file path
    file_path = os.path.join(local_dir, 'data_size_'+str(params['size'])+'_method_'+params['method']+'_id_' + array_id + '_perror_' + str(params['p_error']) + '_psample_' + str(params['p_sampling']) + '.xz')

    # Generate data
    generate(file_path, params, nbr_datapoints=10000)

    # View data file
    
    #iterator = MCMCDataReader(file_path, params['size'])
    #while iterator.has_next():
    #    print('Datapoint nr: ' + str(iterator.current_index() + 1))
    #    print(iterator.next())
