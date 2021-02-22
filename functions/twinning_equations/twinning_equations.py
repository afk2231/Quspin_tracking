import numpy as np
import sys
from functions.original_evolution_equations.evol_eqs import phi as phi_kick


def expiphi(phi):
    return np.exp(-1j * phi)


def expiphiconj(phi):
    return np.exp(1j * phi)


def phi_twin_track(fermihubbard1, fermihubbard2, psi1, psi2):
    """Used for intialization in RH tracking"""
    D1 = fermihubbard1.operator_dict['hop_left_op'].expt_value(psi1)
    D2 = fermihubbard2.operator_dict['hop_left_op'].expt_value(psi2)
    # Define the argument
    arg1 = (D1.imag - D2.imag)
    arg2 = (D1.real - D2.real)
    # Define phi
    phi = np.arctan2(arg1, arg2)
    # Solver is sensitive to whether we specify phi as real or not!
    phi = phi.real
    return phi


def twinning_tracking_evolution(current_time, psi, fermihubbard1, fermihubbard2, perimeter_params, psi_other):

    phi = phi_twin_track(fermihubbard1, fermihubbard2, psi, psi_other)
    # print("ping")

    psi_dot = -expiphi(phi) * perimeter_params.t * fermihubbard1.operator_dict['hop_left_op'].dot(psi)
    psi_dot -= expiphiconj(phi) * perimeter_params.t * fermihubbard1.operator_dict['hop_right_op'].dot(psi)
    psi_dot += fermihubbard1.operator_dict['H_onsite'].dot(psi)
    # sys.exit(0)
    return -1j * psi_dot

def initial_kick_evolution(current_time, psi, fermihubbard, perimeter_params, cycles):
    lat = perimeter_params
    fhm = fermihubbard

    psi_dot = -expiphi(phi_kick(current_time, lat, cycles)) * lat.t * fhm.operator_dict['hop_left_op'].dot(psi)
    psi_dot -= expiphiconj(phi_kick(current_time, lat, cycles)) * lat.t * fhm.operator_dict['hop_right_op'].dot(psi)
    psi_dot += fhm.operator_dict['H_onsite'].dot(psi)

    return -1j * psi_dot

