import numpy as np
import copy
from src.planar_model import Planar_code
from decoders import STDC_general_noise
from src.mwpm import class_sorted_mwpm, regular_mwpm

p_xyz = np.array((0.1, 0.05, 0.0))
size = 11
kwargs = {'p_xyz': p_xyz,
        'p_sampling': 0.3,
        'droplets': 1,
        'steps': size ** 4
        }

code = Planar_code(11)
code.generate_random_error_biased(p_xyz)

mwpm_init = class_sorted_mwpm(code)

ground_state = code.define_equivalence_class()

prediction_state = STDC_general_noise(mwpm_init, **kwargs)

np.set_printoptions(precision=3, suppress=True)
print(ground_state, prediction_state)