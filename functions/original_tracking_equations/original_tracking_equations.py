import numpy as np
import mpmath as mm

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

def phi_J_track_with_branches(perimeter_params, current_time, J_target, fermihubbard, psi, branch_number):
    """Used for intialization in RH tracking"""
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(psi)
    # Define the argument that goes into the arcsin for phi
    arg = -J_target(current_time) / (2 * perimeter_params.a * perimeter_params.t * np.abs(D))
    # Define phi
    phi = ((-1)**branch_number) * np.arcsin(arg + 0j) + np.angle(D) + branch_number * np.pi
    # Solver is sensitive to whether we specify phi as real or not!
    # phi = np.sign(phi.real)*np.abs(phi)
    # phi = phi.real
    alpha = 1
    beta = 1
    if not np.isclose(phi.imag, 0):
        phi = phi.real + alpha * np.sign(phi.imag) * (np.abs(phi.imag))**(beta)
        # phi = np.sign(phi.real) * np.abs(phi)
    else:
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

def original_tracking_evolution_with_branches(current_time, psi, fermihubbard, J_target, perimeter_params, branch_number):

    # tracking_H = - (expiphi(current_time, perimeter_params, J_target, fermihubbard, psi)
    #                 * fermihubbard.operator_dict['hop_left_op']
    #                 + expiphiconj(current_time, perimeter_params, J_target, fermihubbard, psi)
    #                 * fermihubbard.operator_dict['hop_right_op']) + fermihubbard.operator_dict['H_onsite']
    #
    # psi_dot = 1j * tracking_H.dot(psi)
    phi = phi_J_track_with_branches(perimeter_params, current_time, J_target, fermihubbard, psi, branch_number)
    psi_dot = -expiphi(phi) * perimeter_params.t * fermihubbard.operator_dict['hop_left_op'].dot(psi)
    psi_dot -= expiphiconj(phi) * perimeter_params.t * fermihubbard.operator_dict['hop_right_op'].dot(psi)
    psi_dot += fermihubbard.operator_dict['H_onsite'].dot(psi)
    return -1j * psi_dot

def original_tracking_RK4(current_time, psi, fermihubbard, J_target, l, branch_number, t_p):
    k1 = (t_p.delta) * original_tracking_evolution_with_branches(current_time, psi, fermihubbard, J_target, l,
                                                                          branch_number)
    k2 = (t_p.delta) * original_tracking_evolution_with_branches(current_time + (t_p.delta)/2,
                                                                          psi + k1/2, fermihubbard, J_target, l,
                                                                          branch_number)
    k3 = (t_p.delta) * original_tracking_evolution_with_branches(current_time + (t_p.delta)/2,
                                                                          psi + k2/2, fermihubbard, J_target, l,
                                                                          branch_number)
    k4 = (t_p.delta) * original_tracking_evolution_with_branches(current_time + (t_p.delta),
                                                                          psi + k3, fermihubbard, J_target, l,
                                                                          branch_number)
    return psi + (k1 + 2*k2 + 2*k3 + k4)/6
