import numpy as np
from functions.original_tracking_equations.original_tracking_equations import expiphi, expiphiconj
from scipy.optimize import fixed_point


def R_tracking_evolution_equation(current_time, gamma, fermihubbard, observables, J_target):
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])
    comm = fermihubbard.operator_dict['commutator_HK'].expt_value(gamma[:-1])
    R = np.abs(D)
    theta = np.angle(D)
    a = fermihubbard.perimeter_params.a
    t = fermihubbard.perimeter_params.t
    J_dot_target = J_target.derivative()

    R_dot = (comm.real * D.imag - comm.imag * D.real)/R
    # theta_dot = (comm.imag * D.imag + comm.real * D.real)/R**2

    phi = observables.phi_init + theta - gamma[-1]

    psi_dot = -expiphi(phi) * t * fermihubbard.operator_dict['hop_left_op'].dot(gamma[:-1])
    psi_dot -= expiphiconj(phi) * t * fermihubbard.operator_dict['hop_right_op'].dot(gamma[:-1])
    psi_dot += fermihubbard.operator_dict['H_onsite'].dot(gamma[:-1])

    psi_dot = -1j * psi_dot

    y_dot = (J_dot_target(current_time)/(2 * a * t) + R_dot * np.sin(phi - theta))/(R * np.cos(phi - theta))

    gamma_dot = np.append(psi_dot, y_dot)

    return gamma_dot


def H_tracking_evolution_equation(current_time, gamma, fermihubbard, observables, J_target):

    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])

    phi = fixed_point(H_tracking_implicit_phi_function, observables.phi[-1], args=(gamma, J_target(current_time),
                                                                                   fermihubbard, observables))
    # phi = 0

    psi_dot = -expiphi(phi) * fermihubbard.perimeter_params.t \
              * fermihubbard.operator_dict['hop_left_op'].dot(gamma[:-1])
    psi_dot -= expiphiconj(phi) * fermihubbard.perimeter_params.t \
               * fermihubbard.operator_dict['hop_right_op'].dot(gamma[:-1])
    psi_dot += fermihubbard.operator_dict['H_onsite'].dot(gamma[:-1])

    y_dot = -(-fermihubbard.perimeter_params.t * (expiphi(phi) * D
                                                  + expiphiconj(phi) * D.conj())
              + fermihubbard.operator_dict['H_onsite'].expt_value(gamma[:-1]))\
            * (J_target.derivative())(current_time)/(J_target(current_time)**2)

    gamma_dot = np.append(psi_dot, y_dot)

    return gamma_dot


def H_tracking_implicit_phi_function(phi_j0, gamma, J_target, fermihubbard, observables):
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])
    a = fermihubbard.perimeter_params.a
    t = fermihubbard.perimeter_params.t
    expi = expiphi(phi_j0)
    doub = fermihubbard.operator_dict['H_onsite'].expt_value(gamma[:-1])
    phi_0 = observables.phi_init
    bt = observables.boundary_term
    phi_j1 = - a * (-t * (expi * D + (expi * D).conj()) + doub)/J_target + phi_0 + a * gamma[-1] + bt
    return phi_j1.real

def original_R_tracking_evolution_equation(current_time, gamma, fermihubbard, observables, J_target):
    K = fermihubbard.operator_dict['hop_left_op']
    ham_onsite = fermihubbard.operator_dict['H_onsite']
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])
    R_track = np.abs(D)
    theta_track = np.angle(D)
    l = fermihubbard.perimeter_params
    Comm = fermihubbard.operator_dict['commutator_HK'].expt_value(gamma[:-1])
    J_dot_target = J_target.derivative()

    J_dot_tilde = J_dot_target(current_time) / (2 * l.a * l.t)

    R_dot_track = Comm.real * np.sin(theta_track) - Comm.imag * np.cos(theta_track)
    theta_dot_track = (1 / R_track) * (Comm.imag * np.sin(theta_track) + Comm.real * np.cos(theta_track))

    phi_dot = theta_dot_track - (J_dot_tilde - R_dot_track * np.sin(theta_track - gamma[-1])) / (
                R_track * np.cos(theta_track - gamma[-1]))

    psi_dot = -1j * (ham_onsite.dot(gamma[:-1]))
    psi_dot += 1j * l.t * np.exp(-1j * gamma[-1]) * K.dot(gamma[:-1])
    psi_dot += 1j * l.t * np.exp(1j * gamma[-1]) * (K.getH()).dot(gamma[:-1])
    # print(phi_dot)
    # print(psi_dot)

    return np.append(psi_dot, phi_dot.real)
