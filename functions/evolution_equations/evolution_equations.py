import numpy as np
from quspin.operators import commutator
import sys
sys.path.append('.../')
from scipy.optimize import fixed_point

def phi_implicit_H(phi, current_time, gamma, init_cond, perimeter_params, fermihubbard, J_target):
    y_t = gamma[-1]
    psi_t = gamma[:-1]
    phi_0 = init_cond[0]
    bound_con = init_cond[1]

    e_t = fermihubbard.operator_dict['H'].expt_value(psi_t, time=phi)

    return phi_0 - perimeter_params.a * e_t/(J_target(current_time)) + bound_con + perimeter_params.a * y_t

def improved_H_tracking(current_time, gamma, phi_init, init_cond, perimeter_params, fermihubbard, J_dot_target):
    """H tracking equation from the Ehrenfest theorem for the Hamiltonian H"""
    # Separate our extended vector
    y_t = gamma[-1]
    psi_t = gamma[:-1]
    phi_0 = init_cond[0]
    # Fixed point iteration on phi
    J_target = J_dot_target.antiderivative()

    phi_fpi = fixed_point(phi_implicit_H, phi_init, args=(current_time, gamma, init_cond, perimeter_params
                                                          , fermihubbard, J_target))

    # Calculate psi dot

    psi_dot = -1j * fermihubbard.operator_dict['H'].dot(psi_t, time=phi_fpi)

    # Calculate y dot

    y_dot = - fermihubbard.operator_dict['H'].expt_value(psi_t, time=phi_fpi) \
            * J_dot_target(current_time)/((J_target(current_time))**2)

    # recombine for gamma dot

    gamma_dot = np.append(psi_dot, y_dot)

    return gamma_dot

def improved_R_tracking(current_time, gamma, phi_init, init_cond, perimeter_params, fermihubbard, J_dot_target):
    """ R tracking equation from the Ehrenfest theorem for current operator J"""

    # Calculate theta and R from psi

    D = fermihubbard.operator_dict['hop_left_op'].expt_value(gamma[:-1])
    theta_t = np.angle(D)
    R_t = np.abs(D)

    # Get the commutator term

    Comm = commutator(fermihubbard.operator_dict['H_onsite'], fermihubbard.operator_dict['hop_left_op']).expt_value\
        (gamma[:-1])

    # (trivial) fixed point iteration on phi

    phi_fpi = init_cond[0] + theta_t - gamma[-1]

    # Calculate psi dot

    psi_dot = -1j * fermihubbard.operator_dict['H'].dot(gamma[:-1], time=phi_fpi)

    # Calculate y dot

    R_dot = Comm.real * np.sin(theta_t) - Comm.imag * np.cos(theta_t)

    y_dot = (J_dot_target(current_time)/(2*perimeter_params.a*perimeter_params.t) + R_dot * np.sin(theta_t - phi_fpi))\
                  / (R_t * np.cos(theta_t - phi_fpi))

    # recombine for gamma dot

    gamma_dot = np.append(psi_dot, y_dot)

    return gamma_dot


def Hmax_evolution(current_time, gamma, observables, perimeter_params, J_target, fermihubbard, delta):
    # First check if we are below the current tolerance
    if observables.current[-1] < observables.add_var['jtol'] or J_target(current_time) < observables.add_var['jtol']:
        # if we are below the tolerance, we will check the method used in the last step
        if observables.add_var['last_method'][-1] == 'H':
            # if we used H tracking the last step, we will assign the initial conditions for R tracking
            observables.init_cond = np.array([observables.phi[-2], 0])
            observables.y = 0
            # print('switch to R')
        else:
            # if we used R tracking the last step, we will do nothing special
            None

        observables.add_var['last_method'].append('R')

        # evolve our gamma using R tracking
        f_params = dict(phi_init=observables.phi[-1], init_cond=observables.init_cond
                        , perimeter_params=perimeter_params, fermihubbard=fermihubbard
                        , J_dot_target=J_target.derivative())
        gamma_delta_t = RK4(current_time, gamma, delta, improved_R_tracking, **f_params)
        phi_delta_t = observables.init_cond[0] \
                      + np.angle(fermihubbard.operator_dict['hop_left_op'].expt_value(gamma_delta_t[:-1])) \
                      - gamma_delta_t[-1]
        return [phi_delta_t, gamma_delta_t]
    else:
        # if we are above the tolerance, we will check the method used in the last step
        if observables.add_var['last_method'][-1] == 'R':
            # if we used R tracking last step, we will assign the initial conditions for H tracking
            bound_cond = (perimeter_params.a / J_target(current_time)) \
                         * fermihubbard.operator_dict['H'].expt_value(gamma[:-1], time=observables.phi[-1])
            observables.init_cond = np.array([observables.phi[-3], bound_cond])
            gamma[-1] = 0
            # print('switch to H')
        else:
            # if we used R tracking the last step, we will do nothing special
            None

        observables.add_var['last_method'].append('H')

        #evolve our gamma using H tracking

        f_params = dict(phi_init=observables.phi[-4], init_cond=observables.init_cond
                        , perimeter_params=perimeter_params, fermihubbard=fermihubbard
                        , J_dot_target=J_target.derivative())
        gamma_delta_t = RK4(current_time, gamma, delta, improved_H_tracking, **f_params)
        phi_delta_t = fixed_point(phi_implicit_H, observables.phi[-1], args=(current_time + delta, gamma_delta_t
                                                                                  , observables.init_cond
                                                                                  , perimeter_params, fermihubbard
                                                                                  , J_target))
        return [phi_delta_t, gamma_delta_t]


def RK4(current_time, gamma, delta, function, **f_params):
    k1 = delta * function(current_time, gamma, **f_params)
    k2 = delta * function(current_time + 0.5 * delta, gamma + 0.5 * k1, **f_params)
    k3 = delta * function(current_time + 0.5 * delta, gamma + 0.5 * k2, **f_params)
    k4 = delta * function(current_time + delta, gamma + k3, **f_params)

    return gamma + (k1 + 2. * k2 + 2. * k3 + k4) / 6