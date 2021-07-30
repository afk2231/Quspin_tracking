import numpy as np
import mpmath as mm
from scipy.optimize import fixed_point
import sys

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
        # phi = phi.real + alpha * np.sign(phi.real) * (np.abs(phi.imag)) ** (beta)
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
    # print(psi.shape)
    # print(psi.shape)
    # print(psi)
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

def implicit_two_step(psi_np1_0, current_time, psi_n, psi_nm1, t_p, fhm, J_target, l, bn):
    # print(psi_np1_0)
    f_p1 = original_tracking_evolution_with_branches(current_time + t_p.delta, psi_np1_0, fhm, J_target, l, bn)
    f_0 = original_tracking_evolution_with_branches(current_time, psi_n, fhm, J_target, l, bn)
    f_m1 = original_tracking_evolution_with_branches(current_time - t_p.delta, psi_nm1, fhm, J_target, l, bn)
    psi_np1_1 = psi_n + t_p.delta * ((5/12) * f_p1 + (2/3) * f_0 - (1/12) * f_m1)
    return psi_np1_1

def implicit_four_step(psi_np1_0, current_time, psi_n, psi_nm1, psi_nm2, psi_nm3, t_p, fhm, J_target, l, bn):
    # print(psi_np1_0)
    f_p1 = original_tracking_evolution_with_branches(current_time + t_p.delta, psi_np1_0, fhm, J_target, l, bn)
    f_0 = original_tracking_evolution_with_branches(current_time, psi_n, fhm, J_target, l, bn)
    f_m1 = original_tracking_evolution_with_branches(current_time - t_p.delta, psi_nm1, fhm, J_target, l, bn)
    f_m2 = original_tracking_evolution_with_branches(current_time - 2 * t_p.delta, psi_nm2, fhm, J_target, l, bn)
    f_m3 = original_tracking_evolution_with_branches(current_time - 3 * t_p.delta, psi_nm3, fhm, J_target, l, bn)
    psi_np1_1 = psi_n + t_p.delta * ((251/720) * f_p1 + (646/720) * f_0 - (264/720) * f_m1 + (106/720) * f_m2
                                     - (19/720) * f_m3)
    return psi_np1_1


def backwards_diff_four_step(psi_np1_0, current_time, psi_n, psi_nm1, psi_nm2, psi_nm3, t_p, fhm, J_target, l, bn):
    # print(psi_np1_0)
    f_p1 = original_tracking_evolution_with_branches(current_time + t_p.delta, psi_np1_0, fhm, J_target, l, bn)

    psi_np1_1 = (12/25) * t_p.delta * f_p1 + (48/25) * psi_n - (36/25) * psi_nm1 + (16/25) * psi_nm2 - (3/25) * psi_nm3
    return psi_np1_1


def backwards_diff_six_step(psi_np1_0, current_time, psi_n, psi_nm1, psi_nm2, psi_nm3, psi_nm4, psi_nm5, t_p, fhm,
                             J_target, l, bn):
    # print(psi_np1_0)
    f_p1 = original_tracking_evolution_with_branches(current_time + t_p.delta, psi_np1_0, fhm, J_target, l, bn)

    psi_np1_1 = (
            (60/147) * t_p.delta * f_p1 + (360/147) * psi_n - (450/147) * psi_nm1 + (400/147) * psi_nm2
            - (225/147) * psi_nm3 + (72/147) * psi_nm4 - (10/147) * psi_nm5
    )
    return psi_np1_1


def original_tracking_implicit_two_step(current_time, psi, psi_nm1, fhm, J_target, l, bn, tp):
    # print(psi.shape)
    psi_np1 = fixed_point(implicit_two_step,
                          psi,
                          args=(current_time, psi, psi_nm1, tp, fhm, J_target, l, bn)
                          )
    psi_n = psi
    return [psi_np1, psi_n]


def original_tracking_implicit_four_step(current_time, psi, psi_nm1, psi_nm2, psi_nm3, fhm, J_target, l, bn, tp):
    psi_np1 = fixed_point(implicit_four_step,
                          psi,
                          args=(current_time, psi, psi_nm1, psi_nm2, psi_nm3, tp, fhm, J_target, l, bn)
                          )
    psi_n = psi
    return [psi_np1, psi_n, psi_nm1, psi_nm2]


def original_tracking_implicit_bd_four_step(current_time, psi, psi_nm1, psi_nm2, psi_nm3, fhm, J_target, l, bn, tp):
    psi_np1 = fixed_point(backwards_diff_four_step,
                          psi,
                          args=(current_time, psi, psi_nm1, psi_nm2, psi_nm3, tp, fhm, J_target, l, bn)
                          )
    psi_n = psi
    return [psi_np1, psi_n, psi_nm1, psi_nm2]


def original_tracking_implicit_bd_six_step(current_time, psi, psi_nm1, psi_nm2, psi_nm3, psi_nm4, psi_nm5,
                                           fhm, J_target, l, bn, tp):
    psi_np1 = fixed_point(backwards_diff_six_step,
                          psi,
                          args=(
                              current_time,
                              psi,
                              psi_nm1,
                              psi_nm2,
                              psi_nm3,
                              psi_nm4,
                              psi_nm5,
                              tp,
                              fhm,
                              J_target,
                              l,
                              bn
                          ),
                          xtol=1e-14,
                          maxiter=10000
                          )
    psi_n = psi
    return [psi_np1, psi_n, psi_nm1, psi_nm2, psi_nm3, psi_nm4]


def tracking_radau_IIa_5th(ki_0, current_time, psi, fhm, target, l, bn, tp, alpha, beta):
    ki_1 = np.array([np.zeros_like(psi), np.zeros_like(psi), np.zeros_like(psi)])
    for ki, i in zip(ki_0, range(len(ki_0))):
        t_i = current_time + tp.delta * alpha[i]
        psi_i = psi + tp.delta * np.dot(beta[i], ki_0)
        ki_1[i] = tp.delta * original_tracking_evolution_with_branches(t_i, psi_i, fhm, target, l, bn)
    return ki_1


def original_tracking_radau_IIa_5th(current_time, psi, fhm, J_target, l, bn, tp, k1_0, k2_0, k3_0):
    alpha = [(2/5) - np.sqrt(6)/10, (2/5) + np.sqrt(6)/10, 1]
    beta = [
        [(11/45) - 7 * np.sqrt(6)/360, (37/225) - 169 * np.sqrt(6)/1800, -(2/225) + np.sqrt(6)/75],
        [(37/225) + 169 * np.sqrt(6)/1800, (11/45) + 7 * np.sqrt(6)/360, -(2/225) - np.sqrt(6)/75],
        [(4/9) - np.sqrt(6)/36, (4/9) + np.sqrt(6)/36, (1/9)]
    ]

    k1, k2, k3 = fixed_point(
        tracking_radau_IIa_5th,
        np.array([k1_0, k2_0, k3_0]),
        args=(current_time, psi, fhm, J_target, l, bn, tp, alpha, beta)
    )

    return [psi + ((4/9) - np.sqrt(6)/36) * k1 + ((4/9) + np.sqrt(6)/36) * k2 + (1/9) * k3, k1, k2, k3]

def original_tracking_euler(psi_np1_0, psi, current_time, fhm, target, l, bn, tp):

    psi_np1 = psi + tp.delta * original_tracking_evolution_with_branches(
        current_time + tp.delta,
        psi + tp.delta * psi_np1_0,
        fhm,
        target,
        l,
        bn
    )

    return psi_np1

def original_tracking_implicit_euler(current_time, psi, fhm, J_target, l, bn, tp):

    psi_np1 = fixed_point(
        original_tracking_euler,
        psi,
        args=(psi, current_time, fhm, J_target, l, bn, tp),
        xtol=1e-14,
        maxiter=100000
    )

    return psi_np1

