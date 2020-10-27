import numpy as np

def phi(current_time, perimeter_params, cycles):
    phi = (perimeter_params.a * perimeter_params.F0 / perimeter_params.field) \
          * (np.sin(perimeter_params.field * current_time / (2. * cycles)) ** 2.) \
          * np.sin(perimeter_params.field * current_time)
    return phi