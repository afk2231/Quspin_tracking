import sys
sys.path.append('.../')
import numpy as np

class observables:
    def __init__(self, psi_init, J_init, phi_init, perimeter_params, fermihubbard):
        self.fermihubbard = fermihubbard
        self.neighbour = [fermihubbard.operator_dict['hop_left_op'].expt_value(psi_init),]
        self.current = [J_init,]
        self.phi = [phi_init,]
        self.energy = [fermihubbard.operator_dict['ham_init'].expt_value(psi_init),]
        self.y = 0
        self.psi = psi_init
        self.init_cond = np.array([(self.phi)[0], 0])
        self.add_var = dict()

    def append_observables(self, psi, phi):
        self.neighbour.append(self.fermihubbard.operator_dict['hop_left_op'].expt_value(psi))
        self.current.append(self.fermihubbard.operator_dict['tracking_current'].expt_value(psi, time=phi))
        self.energy.append(self.fermihubbard.operator_dict['tracking_H'].expt_value(psi, time=phi))
        self.phi.append(phi)

    def save_observables(self, expectation_dict, method=''):
        expectation_dict["tracking_current" + method] = self.current
        expectation_dict["tracking_phi" + method] = self.phi
        expectation_dict["tracking_neighbour" + method] = self.neighbour
        expectation_dict["tracking_energy" + method] = self.energy
