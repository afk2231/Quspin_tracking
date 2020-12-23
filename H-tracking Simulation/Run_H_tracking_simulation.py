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
from quspin.tools.measurements import obs_vs_time
from classes.fermi_hubbard_class.setup_fermi_hubbard import Fermi_Hubbard
from classes.perimeter_params.tools import parameter_instantiate as hhg  # Used for scaling units.
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params
from classes.observables_class.observables import observables
from functions.original_tracking_equations.original_tracking_equations import phi_J_track
from functions.H_R_tracking_equations.H_R_evol_eqs import H_tracking_evolution_equation, H_tracking_implicit_phi_function
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fixed_point
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
param = unscaledparam(L=6, t0=0.52, U=0., pbc=True, field=32.9, F0=3, a=4, a_scale=1, J_scale=1, tracking=1)

"""Generating our class of scaled parameters. This is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(2e3))

"""load in our initial state."""
infile = './Data/state/state1.npz'
state = dict(np.load(infile))

"""setup quspin operators and lists"""
FHM = Fermi_Hubbard(lat, t_p.cycles)

psi_0 = state['psi_0']

E = FHM.operator_dict["ham_init"].expt_value(psi_0)

psi_t = FHM.operator_dict['ham_init'].evolve(psi_0, 0.0, t_p.times)
psi_t = np.squeeze(psi_t)

expectations = obs_vs_time(psi_t, t_p.times, FHM.operator_dict)

current = -1j * lat.a * lat.t * (expectations['hop_left_op'] - expectations['hop_left_op'].conj())

J_target = UnivariateSpline(t_p.times, param.J_scale * current.real, s=0)

plt.plot(t_p.times * lat.freq, current)

gamma_t = np.append(psi_0, 0)

obs = observables(gamma_t[:-1], J_target(0.0), 0, FHM)

for newtime in tqdm(t_p.times[:-1]):
    solver_args = dict(atol=1e-4)
    gamma_t = evolve(v0=gamma_t, t0=newtime, times=np.array([newtime + t_p.delta]), f=H_tracking_evolution_equation,
                     f_params=[FHM, obs, J_target], solver_name='dopri5', **solver_args)
    gamma_t = gamma_t.reshape(-1)
    phi_t = fixed_point(H_tracking_implicit_phi_function, obs.phi[-1], args=(gamma_t, J_target(newtime),
                                                                                   FHM, obs))
    # phi_t = 0
    obs.append_observables(gamma_t[:-1], phi_t)
    # print(np.abs(obs.neighbour[-1]))
    # print(phi_t)

plt.figure('quick compare')
plt.plot(t_p.times, current)
plt.plot(t_p.times, obs.current, '.')
