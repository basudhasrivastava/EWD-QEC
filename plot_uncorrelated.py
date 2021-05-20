import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os.path
import re

from src.planar_model import Planar_code, _define_equivalence_class
from src.mwpm import regular_mwpm

def success_rates(sizes, p_round_arr):

    success_rate_stdc = np.zeros((8, 32))
    success_rate_mwpm = np.zeros((8, 32))
    #ends = np.ones(8, dtype=int) * 32

    ## general noise error rates
    #p_error = 0.05 + np.arange(32) / 180
    ## rounded error rates for file finding
    #p_error_round = np.round(p_error, 3)
    ## error rates converted to p_x for uncorrelated noise
    #p_error = 1 - np.sqrt(1 - p_error)

    #fixlist = []

    for i, size in enumerate(sizes):
        #path = './output/data_size_{}*_summary.xz'.format(size)
        path = './output/data_size_{}*low_psampling_5L5.xz'.format(size)
        files = glob.glob(path)

        # if there is no data for current size, move to next size
        if not files:
            continue

        # number of p values = number of found files
        #n_probs = len(files)
        #util_code = Planar_code(size)

        for j, p_round in enumerate(p_round_arr):
            # check if file exists, otherwise skip iteration
            #pattern = f'perror_{p_round}(?:add)?(?:_again)?_5L5.xz'
            pattern = f'perror_{p_round}low_psampling_5L5.xz'
            matches = [f for f in files if re.search(pattern, f)]

            if not matches:
            #    ends[i] = min(ends[i], j)
                continue

            #np_list = []
            df_list = []
            n_data = 0
            # Read data
            for file in matches:
                df_tmp = pd.read_pickle(file)
                df_list.append(df_tmp)
                #df_np_tmp = df.to_numpy().ravel()
                #np_list.append(df_np_tmp)
                #n_data += int(df_np_tmp.shape[0] / 2)

            df = pd.concat(df_list, ignore_index=True)
            #df_np = np.concatenate(np_list, axis=0)

            #if n_data < 10000:
            #    fixlist.append([size, j, n_data])

            #success_mwpm = np.zeros(n_data, dtype=bool)
            #success_stdc = np.zeros(n_data, dtype=bool)

            #for k in range(n_data):
            #    qubit_matrix = df_np[2 * k]
            #    true_class = _define_equivalence_class(qubit_matrix)

            #    mwpm_distr = df_np[2 * k + 1][:4]
            #    success_mwpm[k] = (np.argmax(mwpm_distr) == true_class)

            #    stdc_distr = df_np[2 * k + 1][4:]
            #    success_stdc[k] = (np.argmax(stdc_distr) == true_class)

            #success_rate_mwpm[i, j] = np.mean(success_mwpm)
            #success_rate_stdc[i, j] = np.mean(success_stdc)

            #df = pd.read_pickle('./output/data_size_{}_uncorrelated_perror_{:.3f}_summary.xz'.format(size, p_round))
            #df = pd.read_pickle('./output/data_size_{}_uncorrelated_perror_{:.3f}_summary.xz'.format(size, p_round))
            n_data = df.shape[0]

            true_classes = df['qubit_matrix'].map(_define_equivalence_class)
            mwpm_classes = df['mwpm_distr'].map(np.argmax)
            stdc_classes = df['stdc_distr'].map(np.argmax)

            success_rate_mwpm[i, j] = (true_classes == mwpm_classes).mean()
            success_rate_stdc[i, j] = (true_classes == stdc_classes).mean()

            # print number of data points
            print(f'size: {size}, p_x: {p_round:.3f}, n_data: {n_data}, success: {success_rate_stdc[i, j]:.3f}')
    
    np.savez('Success_rates_low_psampling.npz', mwpm=success_rate_mwpm, stdc=success_rate_stdc)

def success_rates_extensive(size, p_round_arr): # single size
    
    success_rate_mwpm = np.zeros(32)
    success_rate_emwpm = np.zeros(32)
    success_rate_stdc_depol = np.zeros(32)
    success_rate_stdc_uncorr = np.zeros(32)

    path = './output/data_size_{}*extensive_5L5.xz'.format(size)
    files = glob.glob(path)

    # if there is no data for current size, move to next size
    if not files:
        return

    for i, p_round in enumerate(p_round_arr):
        # check if file exists, otherwise skip iteration
        pattern = f'perror_{p_round}_extensive_5L5.xz'
        matches = [f for f in files if re.search(pattern, f)]

        if not matches:
            continue

        # Read data, one file at a time
        df_list = []
        for file in matches:
            df_tmp = pd.read_pickle(file)
            df_list.append(df_tmp)

        # Combine list of dataframes into a single dataframe
        df = pd.concat(df_list, ignore_index=True)
        n_data = df.shape[0]

        # Find true class and decoder class for each decoder
        true_classes = df['qubit_matrix'].map(_define_equivalence_class)
        mwpm_classes = df['mwpm_distr'].map(np.argmax)
        emwpm_classes = df['emwpm_distr'].map(np.argmax)
        stdc_depol_classes = df['stdc_depol_samp'].map(np.argmax)
        stdc_uncorr_classes = df['stdc_uncorr_samp'].map(np.argmax)

        success_rate_mwpm[i] = (true_classes == mwpm_classes).mean()
        success_rate_emwpm[i] = (true_classes == emwpm_classes).mean()
        success_rate_stdc_depol[i] = (true_classes == stdc_depol_classes).mean()
        success_rate_stdc_uncorr[i] = (true_classes == stdc_uncorr_classes).mean()

        # print number of data points
        print(f'size: {size}, p_x: {p_round:.3f}, n_data: {n_data}')
    
    np.savez('Success_rates_extensive.npz', mwpm=success_rate_mwpm, emwpm=success_rate_emwpm, \
         stdc_depol=success_rate_stdc_depol, stdc_uncorr=success_rate_stdc_uncorr)


def plot_success(sizes, p_error_arr, success_rate_mwpm, success_rate_stdc):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots(figsize=(8, 6))
    p_x = p_error_arr * (1 - p_error_arr)

    for i in range(len(sizes)):
        ax.plot(p_x, success_rate_mwpm[i], '--', c=colors[i])
        ax.plot(p_x, success_rate_stdc[i], '-', c=colors[i])

        ax.plot([], [], '-', color=colors[i], label=f'{sizes[i]}')

    #print(fixlist)

    ax.plot([], [], '--k', label='mwpm')
    ax.plot([], [], '-k', label='stdc')
    ax.grid(True)
    ax.set_xlabel('p_x')
    ax.set_ylabel('Success rate')
    ax.set_title('p_sampling = 0.1')
    ax.legend()

    fig.savefig('./plots/STDC_MWPM_uncorr_success_low_psampling.png')


def plot_success_extensive(size, p_error_arr, success_rate_mwpm, success_rate_emwpm, \
    success_rate_stdc_depol, success_rate_stdc_uncorr): # single size

    fig, ax = plt.subplots(figsize=(8, 6))
    p_x = p_error_arr * (1 - p_error_arr)

    ax.plot(p_x, success_rate_mwpm, label="MWPM")
    ax.plot(p_x, success_rate_emwpm, label="E-MWPM")
    ax.plot(p_x, success_rate_stdc_depol, label="STDC depolarizing sampling")
    ax.plot(p_x, success_rate_stdc_uncorr, label="STDC uncorrelated sampling")

    ax.grid(True)
    ax.set_xlabel('p_x')
    ax.set_ylabel('success rate')
    ax.set_title('p_sampling = 0.1, L = 19')
    #ax.set_xlim((0.07, 0.11))
    #ax.set_yscale("log")
    ax.legend()

    fig.savefig('./plots/STDC_MWPM_uncorr_success_extensive.png')


def plot_failure(sizes, p_error_arr, success_rate_mwpm, success_rate_stdc):
    # plot log failure rate as funtion of L

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    fig, ax = plt.subplots(figsize=(8, 6))

    logp_fail_mwpm = np.log(1 - success_rate_mwpm)
    logp_fail_stdc = np.log(1 - success_rate_stdc) 

    fig, ax = plt.subplots(figsize=(8, 6))

    for j in range(16, 32, 2):
        p_x = p_error_arr[j] * (1 - p_error_arr[j])
        # find largest L where logp_fail is known and finite
        #tmp = np.nonzero((logp_fail_mwpm[:, j] == 0) | ~np.isfinite(logp_fail_mwpm[:, j]) | \
        #    (logp_fail_stdc[:, j] == 0) | ~np.isfinite(logp_fail_stdc[:, j]))[0]
        #if tmp.size > 0:
        #    L_end = tmp[0]
        #else:
        #    L_end = 8

        ax.plot(sizes, logp_fail_mwpm[:, j], '--', c=colors[j // 2 - 8])
        ax.plot(sizes, logp_fail_stdc[:, j], '-', c=colors[j // 2 - 8], label=rf'$p_x = ${p_x:.3f}')

    ax.plot([], [], '--k', label='mwpm')
    ax.plot([], [], '-k', label='stdc')
    ax.set_xlabel(r'Size, $L$')
    ax.set_ylabel(r'log failure rate, log $P_f$')
    ax.legend()

    fig.savefig('./plots/STDC_MWPM_uncorr_fail_final.png')


def main():
    #sizes = [i for i in range(5, 20, 2)]
    #sizes = [19]
    size=19
    # general noise error rates
    p_error = 0.05 + np.arange(32) / 180
    # rounded error rates for file finding
    p_round_arr = np.round(p_error, 3)
    # error rates converted to p_x for uncorrelated noise
    p_error = 1 - np.sqrt(1 - p_error)

    #success_rates(sizes, p_round_arr)
    #success_rates_extensive(size, p_round_arr)

    #rates = np.load('./Success_rates_low_psampling.npz')
    #success_rate_mwpm = rates['mwpm']
    #success_rate_stdc = rates['stdc']
    #rates.close()
    
    rates = np.load('./Success_rates_extensive.npz')
    success_rate_mwpm = rates['mwpm']
    #print("mwpm: ", success_rate_mwpm)
    success_rate_emwpm = rates['emwpm']
    #print("emwpm: ", success_rate_emwpm)
    success_rate_stdc_depol = rates['stdc_depol']
    success_rate_stdc_uncorr = rates['stdc_uncorr']
    rates.close()

    #plot_success(sizes, p_error, success_rate_mwpm, success_rate_stdc)
    #plot_failure(sizes, p_round_arr, success_rate_mwpm, success_rate_stdc)
    plot_success_extensive(size, p_error, success_rate_mwpm, success_rate_emwpm, success_rate_stdc_depol, success_rate_stdc_uncorr)


if __name__ == '__main__':
    main()