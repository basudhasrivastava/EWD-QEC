from src.toric_model import Toric_code
from src.mcmc import *
from math import *
import numpy as np
import copy
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from matplotlib import rc
#rc('font',**{'family':'sans-serif'})#,'sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif'})#,'serif':['Palatino']})
#rc('text', usetex=True)

def getMCMCstats():
    size = 7
    init_toric = Toric_code(size)
    p_error = 0.1
    Nc = 19
    steps=100000
    
    # define error
    action = Action(position = np.array([1, 1, 0]), action = 3) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 0]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 4, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 0, 0]), action = 1)
    init_toric.step(action)#2
    init_toric.qubit_matrix = np.array([[[0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [0, 1, 2, 1, 1, 0, 0],
                                         [0, 0, 3, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0]],
                                        [[0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 2, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0],
                                         [0, 3, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0]]])
    
    # eller använd någon av dessa för att initiera slumpartat
    #nbr_error = 9
    #init_toric.generate_n_random_errors(10)
    #init_toric.generate_random_error(p_error)
    init_toric.syndrom('next_state')


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    # Start in random eq-class
    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    distr, count, qubitlist = parallel_tempering_plus(init_toric, Nc, p=p_error, steps=steps, iters=10, conv_criteria='error_based')
    print(distr)
    unique_elements, unique_counts = np.unique(qubitlist, axis=0, return_counts=True)
    print('Number of unique elements: ', len(unique_elements))
    #for i in range(len(unique_elements)):
    #    print(np.count_nonzero(unique_elements[i] == qubitlist[:]))
        #unique_counts = np.count_nonzero
    
    # ALL qubit solutions are saved in qubitlist

    shortest = 100
    for i in range(len(qubitlist)):
        nb = np.count_nonzero(qubitlist[i])
        if nb < shortest:
            shortest = nb

    # save all qubits to df_all
    df = pd.DataFrame({"qubit":[], "nbr_err":[], "nbr_occ":[], "eq_class":[]})
    df = pd.concat((pd.DataFrame({"qubit":[unique_elements[i]], "nbr_err":[np.count_nonzero(unique_elements[i])], "nbr_occ":[unique_counts[i]], "eq_class": define_equivalence_class(unique_elements[i])}) for i in range(len(unique_elements))),
            ignore_index=True)
    

    print(df.loc[df['nbr_err'] == shortest])
    print(df.loc[df['nbr_err'] == shortest+1])
    print(df.loc[df['nbr_err'] == shortest+2])
    print(df.loc[df['nbr_err'] == shortest+3])
    print(df.loc[df['nbr_err'] == shortest+4])
    print(df.loc[df['nbr_err'] == shortest+5])
    df2 = df.loc[df['nbr_err'] == shortest].loc[df['eq_class'] == 15]
    i = 0
    for index, row in df2.iterrows():
        i += 1
        #print(row['qubit'])#.reshape((2, 5, 5)))
        init_toric.qubit_matrix = row['qubit']
        init_toric.plot_toric_code(init_toric.next_state, 'Chain_' + str(i))

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def learn_seaborn():
    data = np.transpose(np.arange(32).reshape((2,16))) + 1

    #data = sns.load_dataset("fmri")


    columns = ['p', 'size', 'P', 'std']
    df = pd.DataFrame()
    for i in range(2):
        for j in range(16):
            #tempdf = pd.DataFrame({[[0.05+0.01*j], [i], np.random.randn(1)]})
            tempdf = pd.DataFrame({"p":[0.05+0.01*j], "size": [i], "P":np.random.randn(), "std":sqrt((1-(0.05+0.01*j))*(0.05+0.01*j))})
            print(tempdf)
            df = df.append(tempdf)

    print(df)

    ax = sns.lineplot(x='p', y='P', hue='size', ci='std', data=df)
    #lower_bound = np.arange(16)
    #upper_bound = np.arange(16)+1
    #ax.fill_between([0.05 + 0.01 * i for i in range(16)], lower_bound, upper_bound, alpha=.3)
    #ax.set(xlabel='p', ylabel='P_s')
    #ax.set(xticks=p_s)
    plt.show()


def create_MCMC_df_for_figure_7():
    file_prefix = 'data/'

    correct_guesses = np.zeros((3,16))
    total_guesses = np.zeros((3,16))
    P = np.zeros((3,16))
    x = [0.05 + 0.01*i for i in range(16)] 
    std = np.zeros((3,16))


    #columns = ['p', 'size', 'P']
    df = pd.DataFrame()

    for i in range(3):
        for j in range(16):
            size = (i + 1) * 2 + 1
            p = 0.05 + j * 0.01
            file_suffix = 'data_' + str(size) + 'x' + str(size) + '_p_' + str(p) + '.xz'

            file_path = file_prefix + file_suffix

            #iterator = MCMCDataReader('data/data_000.xz', 5)#file_path, size)
            iterator = MCMCDataReader(file_path, size)
            while iterator.has_next():
                qubit_matrix, eq_distr = iterator.next()
                true_eq = define_equivalence_class(qubit_matrix)
                predicted_eq = np.argmax(eq_distr)
                if predicted_eq == true_eq:
                    correct_guesses[i,j] += 1
                total_guesses[i,j] += 1
            print(j)
            P[i,j] = correct_guesses[i,j]/total_guesses[i,j]
            std[i,j] = sqrt(total_guesses[i,j]*(1-P[i,j])*P[i,j])/total_guesses[i,j]  # Binomial distribution std-deviation
            df = df.append(pd.DataFrame({"p":[p], "d": str(size) + 'x' + str(size), "P":P[i,j], "nbr_pts":total_guesses[i,j], "method":'MCMC'}))
    
    df.to_pickle('data/fig7_MCMC_data.xz')

def create_RL_df_for_figure_7():
    df = pd.DataFrame()
    x = [0.05 + 0.01*i for i in range(16)] 
    P_fusk = [0.9541, 0.9364, 0.9136, 0.8849, 0.8666, 0.8275, 0.7906, 0.7605, 0.7325, 0.6867, 0.6519, 0.6176, 0.5769, 0.5483, 0.5104, 0.4828]
    P = [0.95723333, 0.93936667, 0.91213333, 0.88676667, 0.85806667, 0.82906667, 0.79366667, 0.76386667, 0.72136667, 0.6871,     0.6507,     0.61116667, 0.57353333, 0.54103333, 0.51016667, 0.48093333]
    df = df.append(pd.DataFrame({"p":x, "d": '3x3', "P":P, "nbr_pts":10000, "method":'MCMC+DRL'}))
    #P = 
    #df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'RL_hi'}))
    P = [0.9885, 0.9747, 0.9595, 0.9333, 0.9139, 0.8784, 0.8377, 0.7909, 0.7421, 0.6872, 0.6416, 0.5819, 0.5225, 0.4806, 0.4358, 0.382]
    df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'MCMC+DRL'}))
    df.to_pickle('data/fig7_RL_data.xz')

def create_REF_df_for_figure_7():
    df = pd.DataFrame()
    x = [0.05 + 0.01*i for i in range(16)] 
    
    #P = [0.9556, 0.9411, 0.9169, 0.8878, 0.8639, 0.8192, 0.7868, 0.7547, 0.7243, 0.6789, 0.6392, 0.6053, 0.5703, 0.5286, 0.5016, 0.4625]
    #df = df.append(pd.DataFrame({"p":x, "d": '3x3', "P":P, "nbr_pts":10000, "method":'REF'}))

    P = [0.9909, 0.9814, 0.971, 0.9539, 0.9254, 0.8994, 0.8661, 0.827,  0.7772, 0.7247, 0.6882, 0.6258, 0.587,  0.5278, 0.4868, 0.4274]
    df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'DRL'}))

    P = [0.9981, 0.9946, 0.9887, 0.9787, 0.9595, 0.9358, 0.9031, 0.8692, 0.8129, 0.7582, 0.6983, 0.6327, 0.5652, 0.4965, 0.4379, 0.3814]
    df = df.append(pd.DataFrame({"p":x, "d": '7x7', "P":P, "nbr_pts":10000, "method":'DRL'}))

    #P = [0.9483, 0.9216, 0.8986, 0.8654, 0.8413, 0.8054, 0.7601, 0.7294, 0.7013, 0.654, 0.6191, 0.5898, 0.5508, 0.5109, 0.4754, 0.4468]
    #df = df.append(pd.DataFrame({"p":x, "d": '9x9', "P":P, "nbr_pts":10000, "method":'REF'}))
    df.to_pickle('data/fig7_REF_data.xz')

def create_MWPM_df_for_figure_7():
    df = pd.DataFrame()
    #x = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] 
    x = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2] 

    #P = [9.981500e-01,9.837100e-01,9.497900e-01,9.030200e-01,8.433500e-01,7.753300e-01,7.024700e-01,6.280200e-01,5.537600e-01,4.862800e-01,4.556400e-01]
    P = [9.497900e-01,9.030200e-01,8.433500e-01,7.753300e-01,7.024700e-01,6.280200e-01,5.537600e-01,4.862800e-01,4.556400e-01]
    df = df.append(pd.DataFrame({"p":x, "d": '3x3', "P":P, "nbr_pts":10000, "method":'MWPM'}))

    # lite större MWPM data.
    x = [0.05, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2] 
    P = [9.836900e-01,9.714800e-01,9.279500e-01,8.957500e-01,8.594800e-01,8.165100e-01,7.697700e-01,7.204300e-01,6.712200e-01,6.162400e-01,5.649100e-01,5.143300e-01,4.634000e-01,4.198500e-01,3.721000e-01]
    df = df.append(pd.DataFrame({"p":x, "d": '5x5', "P":P, "nbr_pts":10000, "method":'MWPM'}))
    
    x = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2] 
    P = [9.949200e-01,9.883500e-01,9.765200e-01,9.592800e-01,9.321700e-01,8.990400e-01,8.509200e-01,8.017900e-01,7.396600e-01,6.784800e-01,6.138100e-01,5.487400e-01,4.820700e-01,4.262300e-01,3.697100e-01,3.184300e-01]
    df = df.append(pd.DataFrame({"p":x, "d": '7x7', "P":P, "nbr_pts":10000, "method":'MWPM'}))

    df.to_pickle('data/fig7_MWPM_data.xz')


def plot_fig7_1():
    # Time to plot stuff!
    df = pd.read_pickle('data/fig7_MCMC_data.xz')

    print(df)

    #df = df.append(pd.read_pickle('data/fig7_RL_data.xz'))
    df = df.append(pd.read_pickle('data/fig7_REF_data.xz'))
    #df = df.append(pd.read_pickle('data/fig7_MWPM_data.xz'))

    df.columns = ['p', 'Storlek, $d$', 'P', 'nbr_pts', 'Metod']


    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.lineplot(x='p', y='P', hue='Storlek, $d$', style='Metod', palette=sns.color_palette("cubehelix", 3), data=df)
    #ax.set(xscale="log", yscale="log")
    #ax.set(xlim=(None, 0.1))
    #ax.set(ylim=(0.8, 1))
    
    ax.set_xlabel("Felsannolikhet, $p$",fontsize=15)
    ax.set_ylabel("Korrektionsfrekvens, $P_s$",fontsize=15)
    ax.tick_params(labelsize=12)

    plt.savefig('plots/testing.png')

def plot_fig7_2():
    # Time to plot stuff!
    #df = pd.read_pickle('data/fig7_MCMC_data.xz')


    df = pd.read_pickle('data/fig7_RL_data.xz')
    df = df.append(pd.read_pickle('data/fig7_REF_data.xz'))
    df = df.append(pd.read_pickle('data/fig7_MWPM_data.xz'))

    
    df = df[df.d != '7x7']

    df.columns = ['p', 'Storlek, $d$', 'P', 'nbr_pts', 'Metod']

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    ax = sns.lineplot(x='p', y='P', hue='Storlek, $d$', style='Metod', palette=sns.color_palette("cubehelix", 2), data=df)
    #ax.set(xscale="log", yscale="log")
    #ax.set(xlim=(None, 0.1))
    #ax.set(ylim=(0.8, 1))


    ax.set_xlabel("Felsannolikhet, $p$",fontsize=15)
    ax.set_ylabel("Korrektionsfrekvens, $P_s$",fontsize=15)
    ax.tick_params(labelsize=12)

    plt.show()

def convergence_tester():
    size = 5
    init_toric = Toric_code(size)
    Nc = 9
    p_error = 0.17
    success = 0
    correspondence = 0
    
    for i in range(1000):
          t1 = time.time()
          init_toric.generate_random_error(p_error)
          toric_copy = copy.deepcopy(init_toric)
          apply_random_logical(toric_copy.qubit_matrix)
          class_before = define_equivalence_class(init_toric.qubit_matrix)
          [distr1, _, _, _, _] = parallel_tempering(init_toric, 9, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
          [distr2, _, _, _, _] = parallel_tempering(toric_copy, 9, p=p_error, steps=1000000, iters=10, conv_criteria='error_based')
          class_after = np.argmax(distr1)
          copy_class_after = np.argmax(distr2)
          if class_after == class_before:
              success+=1
          if copy_class_after == class_after:
              correspondence+=1
          
          if i >= 1:
              print('#' + str(i) + " current success rate: ", success/(i+1))
              print('#' + str(i) + " current correspondence: ", correspondence/(i+1), " time: ", time.time()- t1)


def main3(): # P_s som funktion av p
    points = 20
    size = 5
    init_toric = Toric_code(size)
    Nc = 19
    TOPS=20
    SEQ=30
    tops_burn=10
    eps=0.008
    steps=1000000
    p_error = [i*0.01 + 0.05 for i in range(points)]

    # define error
    '''
    action = Action(position = np.array([1, 1, 0]), action = 2) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 3, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 4, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 2]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 3]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 4]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 6, 1]), action = 2)
    init_toric.step(action)#2
    '''

    init_toric.qubit_matrix = np.array([[[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 1],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]],
                                        [[0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0]]])

    #init_toric.generate_n_random_errors(9)

    init_toric.syndrom('next_state')
    


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    t1 = time.time()

    startingqubit = init_toric.qubit_matrix

    data = []

    for i in range(points):
        #init_toric.qubit_matrix, _ = apply_random_logical(startingqubit)

        distr = parallel_tempering(init_toric, Nc=Nc, p=p_error[i], steps=steps, SEQ=SEQ, TOPS=TOPS, tops_burn=tops_burn, eps=eps, conv_criteria='error_based')
        #print("error #" + str(i) + ': ', eq_class_count_BC/np.sum(eq_class_count_BC))
        distr_i = np.divide(distr, np.sum(distr), dtype=np.float)
        data.append(distr_i)
        print(p_error[i], distr_i)
    
    data = np.asarray(data)
    print(data[:,0])
    for i in range(16):
        plt.plot(p_error, data[:,i], label=('eq_class_' + str(i+1)))
    plt.xlabel('Error rate, p')
    plt.ylabel('Probability of equivalance class')
    plt.title('init: k3')
    plt.legend(loc=1)

    plt.show()
        
    print("runtime: ", time.time()-t1)


def eq_evolution():
    size = 5
    init_toric = Toric_code(size)
    p_error = 0.1
    Nc = 15
    steps=10000
    
    # define error
    action = Action(position = np.array([1, 1, 0]), action = 3) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 0]), action = 3)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 4, 0]), action = 1)
    init_toric.step(action)#2
    action = Action(position = np.array([1, 0, 0]), action = 1)
    init_toric.step(action)#2

    # eller använd någon av dessa för att initiera slumpartat
    #nbr_error = 9
    #init_toric.generate_n_random_errors(nbr_error)
    #init_toric.generate_random_error(0.10)
    init_toric.syndrom('next_state')


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init')
    t1 = time.time()

    starting_qubit = init_toric.qubit_matrix

    for i in range(2):
        init_toric.qubit_matrix, _ = apply_random_logical(starting_qubit)

        [distr, eq, eq_full, chain0, resulting_burn_in] = parallel_tempering(init_toric, Nc, p=p_error, steps=steps, iters=10, conv_criteria=None)

        mean_history = np.array([eq[x] / (x + 1) for x in range(steps)])

        plt.plot(mean_history)
        plt.savefig('plots/history_'+str(i+1)+'.png')

    print("runtime: ", time.time()-t1)

def print_qubit():
    size = 5
    init_toric = Toric_code(size)

    # define error
    action = Action(position = np.array([0, 2, 0]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([0, 2, 1]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([0, 2, 2]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([0, 1, 3]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([0, 2, 4]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 2]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1
    action = Action(position = np.array([1, 2, 3]), action = 1) #([vertical=0,horisontal=1, y-position, x-position]), action = x=1,y=2,z=3,I=0)
    init_toric.step(action)#1



    # eller använd någon av dessa för att initiera slumpartat
    #nbr_error = 9
    #init_toric.generate_n_random_errors(nbr_error)
    #init_toric.generate_random_error(0.10)
    init_toric.syndrom('next_state')


    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'bild2_1')
    

def convergence_analysis():
    size = 5
    init_toric = Toric_code(size)
    p_error = 0.185
    Nc = 19
    TOPS=20
    SEQ=30
    tops_burn=10
    eps=0.008
    n_tol=1e-4
    steps=1000000

    criteria = ['error_based'] #, 'distr_based', 'majority_based']

    # define error
    #init_toric.qubit_matrix[1, 1, 0] = 2
    #init_toric.qubit_matrix[1, 2, 0] = 1
    #init_toric.qubit_matrix[1, 3, 0] = 1

    init_toric.qubit_matrix = np.array([[[0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 1, 2, 1, 0],
                                         [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]],
                                        [[0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 2, 0, 0],
                                         [0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0]]])

    # eller använd någon av dessa för att initiera slumpartat
    #nbr_error = 9
    #init_toric.generate_n_random_errors(nbr_error)
    #init_toric.generate_random_error(0.10)
    init_toric.syndrom('next_state')

    # plot initial error configuration
    init_toric.plot_toric_code(init_toric.next_state, 'Chain_init', define_equivalence_class(init_toric.qubit_matrix))
    t1 = time.time()

    init_toric.qubit_matrix, _ = apply_random_logical(init_toric.qubit_matrix)

    [distr, eq, eq_full, chain0, burn_in, crits_distr] = parallel_tempering_analysis(init_toric, Nc, p=p_error, TOPS=TOPS, SEQ=SEQ, tops_burn=tops_burn, eps=eps, n_tol=n_tol, steps=steps, conv_criteria=criteria)

    mean_history = np.array([eq[x] / (x + 1) for x in range(steps)])

    for i in range(16):
        plt.plot(mean_history[: , i], label=i)
    print('Steps to burn in: ', burn_in)
    for crit in criteria:
        print('==============================================')
        print(crit)
        print('convergence step: ', crits_distr[crit][1])
        print('converged distribution: ', crits_distr[crit][0])
        #plt.axvline(x=crits_distr[crit][1], label=crit)

    plt.legend(loc=1)
    plt.show()


if __name__ == '__main__':
    #convergence_tester()
    #eq_evolution()
    #convergence_analysis()
    #main3()
    #the_infamous_figure_7()
    #learn_seaborn()
    #print_qubit()
    #create_RL_df_for_figure_7()
    #create_REF_df_for_figure_7()
    #create_MWPM_df_for_figure_7()
    #create_MCMC_df_for_figure_7()
    #plot_fig7_1()
    #plot_fig7_2()
    getMCMCstats()