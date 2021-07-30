import sys
import copy
sys.path.append('.../')
import numpy as np
from functions.original_tracking_equations.original_tracking_equations import expiphi, expiphiconj
PI = np.pi

class observables:
    # Note: we can alternatively only save phi and psi within this and use the obs_vs_time function given in the quspin
    #   package to calculate any expectation. However, since I had trouble with using phi as a dynamic parameter for
    #   operators in the original tracking strategy, I'll just leave it be for now.
    def __init__(self, psi_init, J_init, phi_init, fermihubbard, continuity=0):
        self.fermihubbard = fermihubbard
        self.psi = [psi_init,]
        self.neighbour = [fermihubbard.operator_dict['hop_left_op'].expt_value(psi_init),]
        self.current = [J_init,]
        self.phi = [phi_init,]
        D = self.neighbour[-1]
        expi = expiphi(phi_init)
        doub = fermihubbard.operator_dict['H_onsite'].expt_value(psi_init)
        a = fermihubbard.perimeter_params.a
        t = fermihubbard.perimeter_params.t
        self.energy = [-t * (D * expi + (D * expi).conj()) + doub,]
        # self.y = 0
        # self.psi = psi_init
        self.phi_init = phi_init
        if np.isclose(J_init, 0):
            self.boundary_term = 0
        else:
            self.boundary_term = a * self.energy[-1]/J_init
        self.number = [self.fermihubbard.perimeter_params.nup + self.fermihubbard.perimeter_params.ndown,]
        self.numbersite = [[], ]
        self.currentsite = [[], ]
        self.continuity = continuity

        if self.continuity:
            for _ in range(self.fermihubbard.perimeter_params.nx - 1):
                self.numbersite.append([])
                self.currentsite.append([])
            for _ in range(self.fermihubbard.perimeter_params.nx - 1):
                self.numbersite[_] = [self.fermihubbard.operator_dict["num"+str(_)].expt_value(psi_init),]
                K_t = expiphi(phi_init) * self.fermihubbard.operator_dict["K" + str(_)].expt_value(psi_init)
                self.currentsite[_] = [-1j * self.fermihubbard.perimeter_params.t * self.fermihubbard.perimeter_params.a
                                       * (K_t - K_t.conj()), ]
            if self.fermihubbard.perimeter_params.pbc:
                self.numbersite[self.fermihubbard.perimeter_params.nx - 1].append(self.fermihubbard.operator_dict["num5"].expt_value(psi_init))
                K_t = expiphi(phi_init) * self.fermihubbard.operator_dict["K5"].expt_value(psi_init)
                self.currentsite[self.fermihubbard.perimeter_params.nx - 1].append(
                    -1j * self.fermihubbard.perimeter_params.t * self.fermihubbard.perimeter_params.a
                    * (K_t - K_t.conj()))
        self.add_var = dict()

    def append_observables(self, psi, phi):
        self.neighbour.append(self.fermihubbard.operator_dict['hop_left_op'].expt_value(psi))
        self.current.append(-1j * self.fermihubbard.perimeter_params.a * self.fermihubbard.perimeter_params.t
                             * (expiphi(phi) * self.neighbour[-1] - expiphiconj(phi) * self.neighbour[-1].conj()))
        self.energy.append(-self.fermihubbard.perimeter_params.t
                           * (expiphi(phi) * self.neighbour[-1] + expiphiconj(phi) * self.neighbour[-1].conj())
                           + self.fermihubbard.operator_dict['H_onsite'].expt_value(psi))
        while abs(phi.real - self.phi[-1].real) > PI:
            if phi.real - self.phi[-1].real > PI:
                phi -= 2 * PI
            elif phi.real - self.phi[-1].real < - PI:
                phi += 2 * PI

        self.phi.append(phi)
        if self.continuity:
            self.number.append(self.fermihubbard.operator_dict['n'].expt_value(psi))
            for _ in range(self.fermihubbard.perimeter_params.nx - 1):
                self.numbersite[_].append(self.fermihubbard.operator_dict["num"+str(_)].expt_value(psi))
                K_t = expiphi(phi) * self.fermihubbard.operator_dict["K" + str(_)].expt_value(psi)
                self.currentsite[_].append(-1j * self.fermihubbard.perimeter_params.t * self.fermihubbard.perimeter_params.a
                                       * (K_t - K_t.conj()))
            if self.fermihubbard.perimeter_params.pbc:
                self.numbersite[self.fermihubbard.perimeter_params.nx - 1].append(self.fermihubbard.operator_dict["num5"].expt_value(psi))
                K_t = expiphi(phi) * self.fermihubbard.operator_dict["K5"].expt_value(psi)
                self.currentsite[self.fermihubbard.perimeter_params.nx - 1].append(
                    -1j * self.fermihubbard.perimeter_params.t * self.fermihubbard.perimeter_params.a
                    * (K_t - K_t.conj()))


    def save_observables(self, expectation_dict, method=''):
        expectation_dict["tracking_current" + method] = self.current
        expectation_dict["tracking_phi" + method] = self.phi
        expectation_dict["tracking_neighbour" + method] = self.neighbour
        expectation_dict["tracking_energy" + method] = self.energy
        if self.continuity:
            expectation_dict["tracking_pnumber" + method] = self.number
            for _ in range(self.fermihubbard.perimeter_params.nx - 1):
                expectation_dict["tracking_pnumbersite" + str(_) + method] = self.numbersite[_]
                expectation_dict["tracking_pcurrentsite" + str(_) + method] = self.currentsite[_]
            if self.fermihubbard.perimeter_params.pbc:
                expectation_dict["tracking_pnumbersite" + str(self.fermihubbard.perimeter_params.nx - 1) + method] = self.numbersite[self.fermihubbard.perimeter_params.nx - 1]
                expectation_dict["tracking_pcurrentsite" + str(self.fermihubbard.perimeter_params.nx - 1) + method] = self.currentsite[self.fermihubbard.perimeter_params.nx - 1]

    def save_observables_branch(self, expectation_dict, branch_string, method=''):
        expectation_dict["tracking_current" + branch_string + method] = self.current
        expectation_dict["tracking_phi" + branch_string + method] = self.phi
        expectation_dict["tracking_neighbour" + branch_string + method] = self.neighbour
        expectation_dict["tracking_energy" + branch_string + method] = self.energy
        if self.continuity:
            expectation_dict["tracking_pnumber" + branch_string + method] = self.number
            for _ in range(self.fermihubbard.perimeter_params.nx - 1):
                expectation_dict["tracking_pnumbersite" + str(_) + branch_string + method] = self.numbersite[_]
                expectation_dict["tracking_pcurrentsite" + str(_) + branch_string + method] = self.currentsite[_]
            if self.fermihubbard.perimeter_params.pbc:
                expectation_dict["tracking_pnumbersite" + str(self.fermihubbard.perimeter_params.nx - 1) + branch_string + method] = self.numbersite[self.fermihubbard.perimeter_params.nx - 1]
                expectation_dict["tracking_pcurrentsite" + str(self.fermihubbard.perimeter_params.nx - 1) + branch_string + method] = self.currentsite[self.fermihubbard.perimeter_params.nx - 1]

    def initialize_work(self):
        work = 0.0
        self.work = [work,]

    def append_work(self, times):
        l = self.fermihubbard.perimeter_params
        if not self.continuity:
            print("Please set continuity in the initialization of observables class to True")
            sys.exit(1)
        phi_dot = np.gradient(self.phi, 1)
        phi_ddot = np.gradient(self.phi, 2)
        work_I = 2 * l.t * phi_dot * np.abs(self.neighbour) * np.cos(np.angle(self.neighbour) - self.phi)

        for n, j in zip(self.numbersite, range(l.nx-1)):
            # print(work_I.shape)
            # print(np.array(n).shape)
            work_I -= phi_ddot * j * np.array(n).real
        # print(work_I)
        work = np.trapz(work_I, times[:len(work_I)])
        # print(work)
        # print(len(self.work))
        self.work.append(work)

    def save_work(self, expectation_dict, method=''):
        print("work expectation saving")
        expectation_dict["tracking_work" + method] = self.work
        print("work expectation saved")

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

    def deepcopy(self):
        obs_deepcopy = observables(
            psi_init=self.psi[0],
            J_init=self.current[0],
            phi_init=self.phi_init,
            fermihubbard=copy.deepcopy(self.fermihubbard),
            continuity=self.continuity
        )
        obs_deepcopy.neighbour = self.neighbour.copy()
        obs_deepcopy.current = self.current.copy()
        obs_deepcopy.energy = self.energy.copy()
        obs_deepcopy.phi = self.phi.copy()
        if self.continuity:
            obs_deepcopy.number = self.number.copy()
            for j in range(self.fermihubbard.perimeter_params.nx - 1):
                obs_deepcopy.numbersite[j] = self.numbersite[j].copy()
                obs_deepcopy.currentsite[j] = self.currentsite[j].copy()
            if self.fermihubbard.perimeter_params.pbc:
                obs_deepcopy.numbersite[self.fermihubbard.perimeter_params.nx - 1] \
                    = self.numbersite[self.fermihubbard.perimeter_params.nx - 1].copy()
                obs_deepcopy.currentsite[self.fermihubbard.perimeter_params.nx - 1] \
                    = self.currentsite[self.fermihubbard.perimeter_params.nx - 1].copy()

        return obs_deepcopy

