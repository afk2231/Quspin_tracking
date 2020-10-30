import numpy as np
from functions.original_tracking_equations.original_tracking_equations import expiphi, expiphiconj
from scipy.optimize import fixed_point


def R_tracking_evolution_equation(current_time, gamma, fermihubbard, observables, J_target):
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])
    comm = fermihubbard.operator_dict['commutator_HK'].expt_value(gamma[:-1])
    R = np.abs(D)
    theta = np.angle(D)

    R_dot = (comm.real * D.imag - comm.imag * D.real)/R
    # theta_dot = (comm.imag * D.imag + comm.real * D.real)/R**2

    phi = observables.phi_init + theta - gamma[-1]

    psi_dot = -expiphi(phi) * fermihubbard.perimeter_params.t \
              * fermihubbard.operator_dict['hop_left_op'].dot(gamma[:-1])
    psi_dot -= expiphiconj(phi) * fermihubbard.perimeter_params.t \
               * fermihubbard.operator_dict['hop_right_op'].dot(gamma[:-1])
    psi_dot += fermihubbard.operator_dict['H_onsite'].dot(gamma[:-1])

    psi_dot = -1j * psi_dot

    y_dot = ((J_target.derivative())(current_time)
             /(2 * fermihubbard.perimeter_params.a * fermihubbard.perimeter_params.t) - R_dot
             * (D.real*np.sin(phi) - D.imag * np.cos(phi))/R)\
            /(D.real * np.cos(phi) + D.imag * np.sin(phi))

    gamma_dot = np.append(psi_dot, y_dot)

    return gamma_dot


def H_tracking_evolution_equation(current_time, gamma, fermihubbard, observables, J_target):

    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])

    phi = fixed_point(H_tracking_implicit_phi_function, observables.phi[-1], args=(gamma, J_target(current_time),
                                                                                   fermihubbard, observables))

    psi_dot = -expiphi(phi) * fermihubbard.perimeter_params.t \
              * fermihubbard.operator_dict['hop_left_op'].dot(gamma[:-1])
    psi_dot -= expiphiconj(phi) * fermihubbard.perimeter_params.t \
               * fermihubbard.operator_dict['hop_right_op'].dot(gamma[:-1])
    psi_dot += fermihubbard.operator_dict['H_onsite'].dot(gamma[:-1])

    y_dot = -(-fermihubbard.perimeter_params.t * (expiphi(phi) * D
                                                  + expiphiconj(phi) * fermihubbard.perimeter_params.t * D.conj())
              + fermihubbard.operator_dict['H_onsite'].expt_value(gamma[:-1]))\
            * (J_target.derivative())(current_time)/(J_target(current_time)**2)

    gamma_dot = np.append(psi_dot, y_dot)

    return gamma_dot


def H_tracking_implicit_phi_function(phi_j0, gamma, J_target, fermihubbard, observables):
    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])
    phi_j1 = - fermihubbard.perimeter_params.a *( -fermihubbard.perimeter_params.t * (expiphi(phi_j0) * D
                                                  + expiphiconj(phi_j0) * fermihubbard.perimeter_params.t * D.conj())
                                                  + fermihubbard.operator_dict['H_onsite'].expt_value(gamma[:-1]))\
             /J_target + observables.phi_init + fermihubbard.perimeter_params.a * gamma[-1] + observables.boundary_term
    return phi_j1