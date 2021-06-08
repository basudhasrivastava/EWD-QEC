import os
import glob

# Import decoders
from decoders import STDC
from decoders_biasednoise import PTEQ

# Import codes
from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.xzzx_model import xzzx_code

# Definitions
codes = {'planar': Planar_code,
         'toric': Toric_code,
         'xzzx': xzzx_code}

decoders = {'PTEQ': PTEQ}

# Settings
# Test related
d = 5  # Code distance
code = codes['planar']  # Code object

decoder = decoders['PTEQ']

# All parameters for all decoders in one place. Refer to the
# decoder source code to see what functions the parameters have
# and which are relevant
decoder_params = {'p_sampling': 0.25,
                  'droplets': 1,
                  'mwpm_init': True,
                  'fixed_errors': None,
                  'Nc': None,
                  'iters': 10,
                  'conv_criteria': 'error_based',
                  'SEQ': 2,
                  'TOPS': 10,
                  'eps': 0.1,
                  'steps': 5 * d ** 4,
                  }

# Biased Z-noise model
error_model = {'p': 0.3,
               'eta': 0.5}

num_syndroms = 2

# Output related
save_syndrom = True  # Save initial syndroms and error chains
clear_old_syndroms = True  # Delete previous syndrom plots
if clear_old_syndroms:
    files = glob.glob('plots/graph_test_*.pdf')
    for file in files:
        os.remove(file)


# Run the test
for syndrom_no in range(1, num_syndroms + 1):
    print(f'Testing syndrom {syndrom_no}/{num_syndroms}')

    # Create code instance
    code_ins = code(size=d)
    code_ins.generate_random_error(error_model['p'],
                                   error_model['eta'])

    # Save the true equivalence class of the syndrom
    eq_true = code_ins.define_equivalence_class()

    if save_syndrom:
        code_ins.plot(f'test_{syndrom_no}', show_eq_class=True)

    eq_distr = decoder(code_ins,
                       error_model['p'],
                       error_model['eta'],
                       **decoder_params)
    print(eq_distr)
