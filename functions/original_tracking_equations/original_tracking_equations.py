import numpy as np

def phi_J_track(perimeter_params, current_time, J_target, fermihubbard, psi):
    """Used for intialization in RH tracking"""
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(psi)
    # Define the argument that goes into the arcsin for phi
    arg = -J_target(current_time) / (2 * perimeter_params.a * perimeter_params.t * np.abs(D))
    # Define phi
    phi = np.arcsin(arg + 0j) + np.angle(D)
    # Solver is sensitive to whether we specify phi as real or not!
    phi = phi.real
    return phi