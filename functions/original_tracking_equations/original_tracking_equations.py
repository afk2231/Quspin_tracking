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

def expiphi(phi):

    return np.exp(-1j * phi)


def expiphiconj(phi):
    return np.exp(1j * phi)

def original_tracking_evolution(current_time, psi, fermihubbard, J_target, perimeter_params):

    # tracking_H = - (expiphi(current_time, perimeter_params, J_target, fermihubbard, psi)
    #                 * fermihubbard.operator_dict['hop_left_op']
    #                 + expiphiconj(current_time, perimeter_params, J_target, fermihubbard, psi)
    #                 * fermihubbard.operator_dict['hop_right_op']) + fermihubbard.operator_dict['H_onsite']
    #
    # psi_dot = 1j * tracking_H.dot(psi)
    phi = phi_J_track(perimeter_params, current_time, J_target, fermihubbard, psi)
    psi_dot = -expiphi(phi) * perimeter_params.t * fermihubbard.operator_dict['hop_left_op'].dot(psi)
    psi_dot -= expiphiconj(phi) * perimeter_params.t * fermihubbard.operator_dict['hop_right_op'].dot(psi)
    psi_dot += fermihubbard.operator_dict['H_onsite'].dot(psi)
    return -1j * psi_dot