import copy
from decoders_biasednoise import PTEQ_biased, PTEQ_alpha, PTEQ_alpha_with_shortest
import os
import sys
import time

import numpy as np
import pandas as pd

from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.xzzx_model import xzzx_code
from src.rotated_surface_model import RotSurCode
from src.mcmc import *
from decoders import *
from src.mwpm import *


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, nbr_datapoints=10**6, fixed_errors=None):

    if params['code'] == 'planar':
        nbr_eq_class = 4
    elif params['code'] == 'toric':
        nbr_eq_class = 16
    elif params['code'] == 'xzzx':
        nbr_eq_class = 4
    elif params['code'] == 'rotated':
        nbr_eq_class = 4
    
    # Creates df
    df = pd.DataFrame()

    # Add parameters to dataframe
    names = ['data_nr', 'type']
    index_params = pd.MultiIndex.from_product([[-1], np.arange(1)],
                                                names=names)
    df_params = pd.DataFrame([[params]],
                            index=index_params,
                            columns=['data'])
    df = df.append(df_params)

    print('\nDataFrame with opened at: ' + str(file_path))


    if fixed_errors != None:
        nbr_datapoints = 10000000
    failed_syndroms = 0

    df_list = []  # Initiate temporary list

    # Loop to generate data points
    for i in range(nbr_datapoints):
        print('Starting generation of point nr: ' + str(i + 1))

        # Initiate code
        if params['code'] == 'toric':
            assert params['noise'] == 'depolarizing'
            init_code = Toric_code(params['size'])
            init_code.generate_random_error(params['p_error'])
        elif params['code'] == 'planar':
            assert params['noise'] in ['depolarizing', 'alpha']
            init_code = Planar_code(params['size'])
            if params['noise'] == 'depolarizing':
                init_code.generate_random_error(params['p_error']/3, params['p_error']/3, params['p_error']/3)
            elif params['noise'] == 'alpha':
                pz_tilde = params['p_error']
                alpha = params['alpha']
                
                p_tilde = pz_tilde + 2*pz_tilde**alpha
                p = p_tilde / (1 + p_tilde)
                p_z = pz_tilde*(1 - p)
                p_x = p_y = pz_tilde**alpha * (1 - p)
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        elif params['code'] == 'xzzx':
            init_code = xzzx_code(params['size'])
            if params['noise'] == 'biased':
                eta = params['eta']
                p = params['p_error']
                p_z = p * eta / (eta + 1)
                p_x = p / (2 * (eta + 1))
                p_y = p_x
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
            if params['noise'] == 'alpha':
                pz_tilde = params['p_error']
                alpha = params['alpha']
                
                p_tilde = pz_tilde + 2*pz_tilde**alpha
                p = p_tilde / (1 + p_tilde)
                p_z = pz_tilde*(1 - p)
                p_x = p_y = pz_tilde**alpha * (1 - p)
                
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
            if params['noise'] == 'depolarizing':
                p_x = p_y = p_z = params['p_error']/3
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        elif params['code'] == 'rotated':
            init_code = RotSurCode(params['size'])
            if params['noise'] == 'biased':
                eta = params['eta']
                p = params['p_error']
                p_z = p * eta / (eta + 1)
                p_x = p / (2 * (eta + 1))
                p_y = p_x
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
            if params['noise'] == 'alpha':
                pz_tilde = params['p_error']
                alpha = params['alpha']
                
                p_tilde = pz_tilde + 2*pz_tilde**alpha
                p = p_tilde / (1 + p_tilde)
                p_z = pz_tilde*(1 - p)
                p_x = p_y = pz_tilde**alpha * (1 - p)
                
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
            if params['noise'] == 'depolarizing':
                p_x = p_y = p_z = params['p_error']/3
                init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
 
        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix)
        eq_true = init_code.define_equivalence_class()


        
        if params['mwpm_init']: #get mwpm starting points
            assert params['code'] == 'planar'
            init_code = class_sorted_mwpm(init_code)
            print('Starting in MWPM state')
        else: #randomize input matrix, no trace of seed.
            init_code.qubit_matrix, _ = init_code.apply_random_logical()
            #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            print('Starting in random state')

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if params['method'] == "PTEQ":
            if params['noise'] == 'depolarizing':
                df_eq_distr = PTEQ(init_code, params['p_error'])
                if np.argmax(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            if params['noise'] == "biased":

                p = params['p_error']
                eta = params['eta']

                pz_tilde = (p / (1 + 1/eta)) / (1-p)
                alpha = np.log(pz_tilde/(2*eta)) / np.log(pz_tilde)

                df_eq_distr = PTEQ_alpha(init_code, pz_tilde, alpha=alpha)
                if np.argmax(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            if params['noise'] == "alpha":
                df_eq_distr = PTEQ_alpha(init_code, params['p_error'], alpha=params['alpha'])
                if np.argmax(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
        if params['method'] == "PTEQ_with_shortest":
            assert params['noise'] == 'alpha'
            if params['noise'] == "alpha":
                df_eq_distr = PTEQ_alpha_with_shortest(init_code, params['p_error'], alpha=params['alpha'])
                if np.argmax(df_eq_distr[0:4]) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
        if params['method'] == "PTDC":
            df_eq_distr, conv = PTDC(init_code, params['p_error'], params['p_sampling'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        if params['method'] == "PTRC":
            df_eq_distr, conv = PTRC(init_code, params['p_error'], params['p_sampling'])
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STDC":
            df_eq_distr = STDC(init_code, params['p_error'], params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STDC_N_n":
            assert params['noise'] == 'alpha'
            df_eq_distr = STDC_Nall_n_alpha(init_code,
                               params['p_sampling'],
                               params['alpha'],
                               params['p_error'],
                               steps=params['steps'])
            df_eq_distr = np.array(df_eq_distr)
        elif params['method'] == "ST":
            df_eq_distr = single_temp(init_code, params['p_error'], params['steps'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmin(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "STRC":
            df_eq_distr = STRC(init_code, params['p_error'], p_sampling=params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
            df_eq_distr = np.array(df_eq_distr)
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "eMWPM":
            out = class_sorted_mwpm(copy.deepcopy(init_code))
            lens = np.zeros((4))
            for j in range(4):
                lens[j] = sum(out[j].chain_lengths())
            choice = np.argmin(lens)
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "MWPM":
            choice = regular_mwpm(copy.deepcopy(init_code))
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1

        # Generate data for DataFrame storage  OBS now using full bincount, change this

        # Create indices for generated data
        names = ['data_nr', 'type']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(1)],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(1)+1], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame([[df_qubit.astype(np.uint8)]], index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame([[df_eq_distr]],
                                index=index_distr, columns=['data'])

        # Add dataframes to temporary list to shorten computation time
        
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list
        
        if (i + 1) % 50 == 0: # this needs to be sufficiently big that rsync has time to sync files before update, maybe change this to be time-based instead.
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)
            print('Failed so far:', failed_syndroms)
        
        # If the desired amount of errors have been achieved, break the loop and finish up
        if failed_syndroms == fixed_errors:
            print('Desired amount of failes syndroms achieved, breaking loop.')
            break

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)
    
    print('\nCompleted')


if __name__ == '__main__':
    # Get job array id, working directory
    job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    local_dir = os.getenv('TMPDIR')

    params = {'code': "xzzx",
            'method': "PTEQ",
            'size': 7,
            'noise': 'alpha',
            'p_error': np.round((0.01 + float(array_id) / 50), decimals=2),
            'eta': 0.5,
            'alpha': 1,
            'p_sampling': 0.27,#np.round((0.01 + float(array_id) / 50), decimals=2),
            'droplets': 1,
            'mwpm_init': False,
            'fixed_errors':None,
            'Nc':None,
            'iters': 10,
            'conv_criteria': 'error_based',
            'SEQ': 2,
            'TOPS': 10,
            'eps': 0.1/2}
    # Steps is a function of code size L
    params.update({'steps': int(5*params['size']**5)})

    print('Nbr of steps to take if applicable:', params['steps'])
    
    with open('/cephyr/users/hamkarl/Vera/MCMC-QEC-toric-RL/generate_data.py', 'r') as f:
        print(f.read(), flush=True)
    with open('/cephyr/users/hamkarl/Vera/MCMC-QEC-toric-RL/decoders.py', 'r') as f:
        print(f.read(), flush=True)
    with open('/cephyr/users/hamkarl/Vera/MCMC-QEC-toric-RL/decoders_biasednoise.py', 'r') as f:
        print(f.read(), flush=True)

    # Build file path
    file_path = os.path.join(local_dir, 'data_d7-3aiii_' + job_id + '_' + array_id + '.xz')
    # Generate dataii
    generate(file_path, params, nbr_datapoints=10000, fixed_errors=params['fixed_errors'])

    # View data file
    
    # iterator = MCMCDataReader(file_path, params['size'])
    # data = iterator.full()
    # for k in range(int(len(data)/2)):
    #     qubit_matrix = data[2*k]#.reshape(2,params['size'],params['size'])
    #     eq_distr = data[2*k+1]

    #     print(qubit_matrix)
    #     print(eq_distr)
