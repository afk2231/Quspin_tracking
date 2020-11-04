from quspin.basis import spinful_fermion_basis_1d  # Hilbert space basis
from quspin.operators import hamiltonian, commutator
from functions.original_evolution_equations.evol_eqs import phi
from functions.original_tracking_equations.original_tracking_equations import expiphi as expiphi_T
from functions.original_tracking_equations.original_tracking_equations import expiphiconj as expiphiconj_T
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
        self.hop_right = [[perimeter_params.t, i, i + 1] for i in range(perimeter_params.nx - 1)]  # hopping to the right OBC
        self.hop_left = [[-perimeter_params.t, i, i + 1] for i in range(perimeter_params.nx - 1)]  # hopping to the left OBC
        if perimeter_params.pbc:
            self.hop_right.append([perimeter_params.t, perimeter_params.nx - 1, 0])
            self.hop_left.append([-perimeter_params.t, perimeter_params.nx - 1, 0])
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        hop_left_op = hamiltonian([["+-|", self.hop_left], ["|+-", self.hop_left]], [], basis=self.basis,
                                  **no_checks)  # left hoperators
        hop_right_op = hop_left_op.getH()  # right hoperators
        ham_onsite = hamiltonian(self.sHl, [], basis=self.basis)  # here we just use the onsite Hamiltonian
        self.operator_dict = dict(H_onsite=ham_onsite)
        self.operator_dict["hop_left_op"] = -hop_left_op / perimeter_params.t
        self.operator_dict["hop_right_op"] = -hop_right_op / perimeter_params.t
        self.operator_dict["ham_init"] = ham_onsite + (hop_left_op + hop_right_op)

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
        self.perimeter_params = perimeter_params

    def create_commutator(self):
        self.operator_dict['commutator_HK'] = commutator(self.operator_dict['H_onsite'],
                                                         self.operator_dict['hop_left_op'])

    def create_number_operator(self):
        num_list = [[1, _] for _ in range(self.perimeter_params.nx)]
        self.operator_dict['n'] = hamiltonian([["n|", num_list], ["|n", num_list]], [], basis=self.basis)
        for _ in range(self.perimeter_params.nx):
            self.operator_dict['num'+str(_)] = hamiltonian([["n|", [[1, _], ]], ["|n", [[1, _], ]]], [],
                                                           basis=self.basis)

    def create_current_operator(self):
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        for _ in range(self.perimeter_params.nx - 1):
            self.operator_dict["K"+str(_)] = hamiltonian([["+-|", [[1, _, _ + 1], ]], ["|+-", [[1, _, _ + 1], ]]],
                                                         [], basis=self.basis, **no_checks)
        if self.perimeter_params.pbc:
            self.operator_dict["K"+str(self.perimeter_params.nx - 1)] = hamiltonian(
                [["+-|", [[1, self.perimeter_params.nx - 1, 0], ]], ["|+-", [[1, self.perimeter_params.nx - 1, 0], ]]],
                [], basis=self.basis, **no_checks)
