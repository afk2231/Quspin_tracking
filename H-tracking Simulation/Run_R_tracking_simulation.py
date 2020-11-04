###############################################################################
#   H-tracking-esque simulation. Utilizes tracking strategies derived from
#   Ehrenfest theorems. This simulation purely uses the method derived from
#   the Ehrenfest theorem for the current operator J, R-tracking.
#   Used to
###############################################################################

from __future__ import print_function, division
import os
import sys
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
from classes.fermi_hubbard_class.setup_fermi_hubbard import Fermi_Hubbard
from classes.perimeter_params.tools import parameter_instantiate as hhg  # Used for scaling units.
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params
from classes.observables_class.observables import observables
from functions.original_tracking_equations.original_tracking_equations import phi_J_track
from functions.H_R_tracking_equations.H_R_evol_eqs import R_tracking_evolution_equation
from scipy.interpolate import UnivariateSpline
import psutil
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from tqdm import tqdm
from matplotlib import pyplot as plt

"""Open MP and MKL should speed up the time required to run these simulations!"""
threads = 6
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
sys.path.append('../')
# note cpu_count for logical=False returns the wrong number for multi-socket CPUs.
print("logical cores available {}".format(psutil.cpu_count(logical=True)))
t_init = time()
np.__config__.show()

"""Generate our class for the unscaled parameters. These are primarily used for saving our data"""
param = unscaledparam(L=6, t0=0.52, U=1, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

"""Generating our class of scaled parameters. This is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(2e4))

"""importing our data from preliminary simulation"""
loadfile = '../Preliminary simulation/Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'\
    .format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc)
expectations = dict(np.load(loadfile))

J_target = UnivariateSpline(t_p.times, param.J_scale * expectations['current'], s=0)

# plt.figure('current')
# plt.plot(t_p.times, J_target(t_p.times))
# plt.plot(t_p.times, expectations['current'], ".")

"""setup quspin operators and lists"""
FHM = Fermi_Hubbard(lat, t_p.cycles)

"""get inital energy and state"""
ti = time()
E, psi_0 = FHM.operator_dict['ham_init'].eigsh(k=1, which='SA')
gamma_t = np.append(np.squeeze(psi_0), 0.0)
print("Initial state and energy calculated. It took {:.2f} seconds to calculate".format(time() - ti))
print("Ground state energy was calculated to be {:.2f}".format(E[0]))

"""create number operator"""

FHM.create_number_operator()

"""create current operator"""

FHM.create_current_operator()

"""create our observables class"""
obs = observables(gamma_t[:-1], J_target(0.0), phi_J_track(lat, 0.0, J_target, FHM, gamma_t[:-1]), FHM)

"""create our commutator operator with the hamiltonian H and the left hopping operator K"""

FHM.create_commutator()

"""Start our evolution"""

for newtime in tqdm(t_p.times[:-1]):
    solver_args = dict(atol=1e-4)
    gamma_t = evolve(v0=gamma_t, t0=newtime, times=np.array([newtime + t_p.delta]), f=R_tracking_evolution_equation,
                     f_params=[FHM, obs, J_target], solver_name='dopri5', **solver_args)
    gamma_t = gamma_t.reshape(-1)
    phi_t = obs.phi_init + np.angle(FHM.operator_dict['hop_left_op'].expt_value(gamma_t[:-1])) - gamma_t[-1]
    obs.append_observables(gamma_t[:-1], phi_t)
    # print(np.abs(obs.neighbour[-1]))
    # print(phi_t)

"""save all observables to a file"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}' \
          '-method={}.npz'.format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps,
                                  param.pbc, param.a_scale, param.J_scale, 'R_tracking')

obs.save_observables(expectations, method='_R_tracking')
print('Saving our expectations.')
ti = time()
np.savez(outfile, **expectations)
print('Expectations saved. It took {:.2f} seconds.'.format(time() - ti))

print('Program finished. It took {:.2f} seconds to run'.format(time() - t_init))