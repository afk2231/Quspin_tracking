from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.operators import hamiltonian
from functions.original_evolution_equations.evol_eqs import phi
import numpy as np

def expiphi(current_time, perimeter_params, cycles):

    return np.exp(-1j * phi(current_time, perimeter_params=perimeter_params, cycles=cycles))


def expiphiconj(current_time, perimeter_params, cycles):
    return np.exp(1j * phi(current_time, perimeter_params=perimeter_params, cycles=cycles))


class Fermi_Hubbard:
    def __init__(self, perimeter_params, cycles):
        self.basis = spinful_fermion_basis_1d(perimeter_params.nx, Nf=(perimeter_params.nup, perimeter_params.ndown))
        self.int_list = [[perimeter_params.U, i, i] for i in range(perimeter_params.nx)]
        self.sHl = [["n|n", self.int_list],]  # onsite interaction
        self.hop_right = [[-perimeter_params.t, i, i + 1] for i in range(perimeter_params.nx - 1)]  # hopping to the right OBC
        self.hop_left = [[perimeter_params.t, i, i + 1] for i in range(perimeter_params.nx - 1)]  # hopping to the left OBC
        if perimeter_params.pbc:
            self.hop_right.append([-perimeter_params.t, perimeter_params.nx - 1, 0])
            self.hop_left.append([perimeter_params.t, perimeter_params.nx - 1, 0])
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        hop_left_op = hamiltonian([["+-|", self.hop_left], ["|+-", self.hop_left]], [], basis=self.basis,
                                  **no_checks)  # left hoperators
        hop_right_op = hop_left_op.getH()  # right hoperators
        ham_onsite = hamiltonian(self.sHl, [], basis=self.basis)  # here we just use the onsite Hamiltonian
        self.operator_dict = dict(H_onsite=ham_onsite)
        self.operator_dict["hop_left_op"] = hop_left_op
        self.operator_dict["hop_right_op"] = hop_right_op
        self.operator_dict["ham_init"] = ham_onsite - (hop_left_op + hop_right_op)

        dynamic_args = [perimeter_params, cycles]
        self.dHl = [
            ["+-|", self.hop_left, expiphi, dynamic_args],  # up hop left
            ["-+|", self.hop_right, expiphiconj, dynamic_args],  # up hop right
            ["|+-", self.hop_left, expiphi, dynamic_args],  # down hop left
            ["|-+", self.hop_right, expiphiconj, dynamic_args],  # down hop right
        ]
        ham = hamiltonian(self.sHl, self.dHl, basis=self.basis)

        self.operator_dict["H"] = ham
        self.operator_dict["lhopup"] = hamiltonian([], [["+-|", self.hop_left, expiphi, dynamic_args]], basis=self.basis
                                                   , **no_checks) / perimeter_params.t
        self.operator_dict["lhopdown"] = hamiltonian([], [["|+-", self.hop_left, expiphi, dynamic_args]], basis=self.basis,
                                                **no_checks) / perimeter_params.t
        self.operator_dict["current"] = 1j * perimeter_params.a * perimeter_params.t * ((self.operator_dict["lhopup"]
                                                                    + self.operator_dict["lhopdown"])
                                                                   - (self.operator_dict["lhopup"]
                                                                      + self.operator_dict["lhopdown"]).getH())
    def create_tracking_hamiltonian(self,dynamic_args, expiphi, expiphiconj):
        dHl = [
            ["+-|", self.hop_left, expiphi, dynamic_args],  # up hop left
            ["-+|", self.hop_right, expiphiconj, dynamic_args],  # up hop right
            ["|+-", self.hop_left, expiphi, dynamic_args],  # down hop left
            ["|-+", self.hop_right, expiphiconj, dynamic_args],  # down hop right
        ]
        self.operator_dict['tracking_H'] = hamiltonian(self.sHl, dHl, basis=self.basis)
