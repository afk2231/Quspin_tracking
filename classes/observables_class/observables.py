import sys
sys.path.append('.../')
import numpy as np
from functions.original_tracking_equations.original_tracking_equations import expiphi, expiphiconj
PI = np.pi

class observables:
    # Note: we can alternatively only save phi and psi within this and use the obs_vs_time function given in the quspin
    #   package to calculate any expectation. However, since I had trouble with using phi as a dynamic parameter for
    #   operators in the original tracking strategy, I'll just leave it be for now.
    def __init__(self, psi_init, J_init, phi_init, fermihubbard):
        self.fermihubbard = fermihubbard
        self.neighbour = [fermihubbard.operator_dict['hop_left_op'].expt_value(psi_init),]
        self.current = [J_init,]
        self.phi = [phi_init,]
        self.energy = [fermihubbard.operator_dict['ham_init'].expt_value(psi_init),]
        # self.y = 0
        # self.psi = psi_init
        self.phi_init = phi_init
        self.boundary_term = 0.0
        self.add_var = dict()

    def append_observables(self, psi, phi):
        self.neighbour.append(self.fermihubbard.operator_dict['hop_left_op'].expt_value(psi))
        self.current.append(-1j * self.fermihubbard.perimeter_params.a * self.fermihubbard.perimeter_params.t
                             * (expiphi(phi) * self.neighbour[-1] - expiphiconj(phi) * self.neighbour[-1].conj()))
        self.energy.append(-self.fermihubbard.perimeter_params.t
                           * (expiphi(phi) * self.neighbour[-1] + expiphiconj(phi) * self.neighbour[-1].conj())
                           + self.fermihubbard.operator_dict['H_onsite'].expt_value(psi))
        self.phi.append(phi)

    def save_observables(self, expectation_dict, method=''):
        expectation_dict["tracking_current" + method] = self.current
        expectation_dict["tracking_phi" + method] = self.phi
        expectation_dict["tracking_neighbour" + method] = self.neighbour
        expectation_dict["tracking_energy" + method] = self.energy

    def switch_tracking_methods(self, method, fermihubbard, psi, J_target):
        if method == "R2H":
            D = fermihubbard.operator_dict['hop_left_op'].expt_value(psi)
            self.phi_init = self.phi[-1]
            self.boundary_term = fermihubbard.perimeter_params.a \
                                 * (-fermihubbard.perimeter_params.t * (expiphi(self.phi_init) * D
                                                                        + expiphiconj(self.phi_init)
                                                                        * fermihubbard.perimeter_params.t * D.conj())
                                     + fermihubbard.operator_dict['H_onsite'].expt_value(psi))/J_target
        elif method == "H2R":
            self.phi_init = self.phi[-1]
            self.boundary_term = 0.0

        return 0.0