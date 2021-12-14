import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from src.xzzx_model import _define_equivalence_class as define_eq_xzzx
from math import sqrt

file_prefix = 'data/EWD_xzzx5_alpha1_'

file_path_list = sorted(glob(file_prefix + '*'),
    key=lambda a: float(a.split("_")[3][:-3]))

df = pd.DataFrame()

for file_path in file_path_list:

    data = pd.read_pickle(file_path).to_numpy().ravel()

    # Extract paramters and data from dataframe
    params = data[0]
    data = np.delete(data, 0)

    p_error = params['p_error']
    size = params['size']
    model = params['code']

    total = 0
    failed = 0

    for k in range(int(len(data)/2)):
        qubit_matrix = data[2*k].reshape(size, size)
        eq_distr = data[2*k+1]

        true_eq = define_eq_xzzx(qubit_matrix)
        
        predicted_eq = np.argmax(eq_distr[:4])

        total += 1
        if true_eq != predicted_eq:
            failed += 1

    P_e = failed/total
    df = df.append(pd.DataFrame({"p": [p_error], "d": size, "P_e": P_e,
        "nbr_pts": int(len(data)/2), "method": params['method']}))



df['std'] = df['P_e']*(1 - df['P_e'])/df['nbr_pts']
df['std'] = df['std'].apply(lambda x: sqrt(x))

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot()

ps = df['p'].to_numpy()
P_es = df['P_e'].to_numpy()
err = df['std'].to_numpy()

ax.errorbar(ps, P_es, err, capsize=2)

ax.set_xlabel("Physical error rate, $p$")
ax.set_ylabel("Logical fail rate, $P_f$")

plt.tight_layout()
plt.savefig('plots/example.pdf')
print(df)