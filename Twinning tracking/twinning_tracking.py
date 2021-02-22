###############################################################################
#   Basic simulation used to generate data for twinning tracking.
#   Starts by perturbing the ground state of a system.
###############################################################################

from __future__ import print_function, division
import os
import sys
import shutil
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
from classes.fermi_hubbard_class.setup_fermi_hubbard import Fermi_Hubbard
from classes.perimeter_params.tools import parameter_instantiate as hhg  # Used for scaling units.
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params
from classes.observables_class.observables import observables
from functions.original_tracking_equations.original_tracking_equations import expiphi, expiphiconj, phi_J_track\
    , original_tracking_evolution
from functions.twinning_equations.twinning_equations import twinning_tracking_evolution, phi_twin_track, initial_kick_evolution
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

ti = time()
print("Building initial classes")
"""Generate our class for the unscaled parameters. These are primarily used for saving our data"""
param = unscaledparam(L=6, t0=0.52, U=0.001, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

"""Generating our class of scaled parameters. This is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""Generate our class for the unscaled parameters for the twinned system."""
param2 = unscaledparam(L=6, t0=0.52, U=1, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

"""Generating our class of scaled parameters for the twinned system."""
lat2 = hhg(field=param2.field, nup=param2.N_up, ndown=param2.N_down, nx=param2.L, ny=0, U=param2.U, t=param2.t0, F0=param2.F0
          , a=param2.a, pbc=param2.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=10, nsteps=int(2e3))

# plt.figure('current')
# plt.plot(t_p.times, J_target(t_p.times))
# plt.plot(t_p.times, expectations['current'], ".")

"""setup quspin operators and lists"""
FHM1 = Fermi_Hubbard(lat, t_p.cycles)
FHM2 = Fermi_Hubbard(lat2, t_p.cycles)

print("Initial classes built. This took {:.2f} seconds".format(time() - ti))

"""get inital energy and state"""
print("Determining ground states of the two systems.")
ti = time()
E1, psi1_0 = FHM1.operator_dict['ham_init'].eigsh(k=1, which='SA')
psi1_0 = np.squeeze(psi1_0)
E2, psi2_0 = FHM2.operator_dict['ham_init'].eigsh(k=1, which='SA')
psi2_0 = np.squeeze(psi2_0)
print("Ground states determined. This took {:.2f}".format(time() - ti))

"""kick one of the systems for the initial state."""
print("Generating Initial states by kicking the ground state of the two systems.")
ti = time()
t_p_kick = time_evolution_params(perimeter_params=lat, cycles=10, nsteps=int(2e3))
psi1_init = psi1_0.copy()
psi2_init = psi2_0.copy()

for current_time in tqdm(t_p_kick.times[:int(len(t_p_kick.times)/3.2)]):
    solver_args = dict(atol=1e-12)
    # print(psi_t.shape)
    psi1_init = evolve(v0=psi1_init, t0=current_time, times=np.array([current_time + t_p.delta]),
                        f=initial_kick_evolution, f_params=[FHM1, lat, t_p_kick.cycles], **solver_args)
    psi2_init = evolve(v0=psi2_init, t0=current_time, times=np.array([current_time + t_p.delta]),
                        f=initial_kick_evolution, f_params=[FHM2, lat2, t_p_kick.cycles], **solver_args)
    psi1_init = np.squeeze(psi1_init)
    psi2_init = np.squeeze(psi2_init)

# psi1_init = FHM1.operator_dict["H"].evolve(psi1_0, 0.0, t_p_kick.times[:int(len(t_p_kick.times)/5)])
# psi1_init = np.squeeze(psi1_init)
# psi2_init = FHM2.operator_dict["H"].evolve(psi2_0, 0.0, t_p_kick.times[:int(len(t_p_kick.times)/5)])
# psi2_init = np.squeeze(psi2_init)

"""extract initial states"""
psi1_t = psi1_init.copy()
psi2_t = psi2_init.copy()
print(psi1_t.shape)
if not (np.isclose(np.linalg.norm(psi1_t), 1) and np.isclose(np.linalg.norm(psi2_t), 1)):
    print("Initial states are not normalized.")
    sys.exit(1)
else:
    print("Initial states are normalized")
    # sys.exit(0)

if np.allclose(psi1_t, psi1_0) or np.allclose(psi2_t, psi2_0):
    print("Initial states are the same as the ground states.")
    sys.exit(1)
else:
    print("Initial states are different from the ground states.")
    # sys.exit(0)

if np.allclose(psi1_t, psi2_t):
    print("Initial states are the same.")
    sys.exit(1)
else:
    print("Initial states are not the same")
    # sys.exit(0)

# if not (np.allclose(psi1_t, psi1_0) and np.allclose(psi2_t, psi2_0)):
#     print("Ground states are not the same after extraction")
#     sys.exit(1)
# else:
#     print("Ground states are the same after extraction")
#     sys.exit(0)
# print(psi1_t)
print("Initial states generated through kicking ground state. This took {:.2f} seconds.".format(time() - ti))

print("Building more classes.")
ti = time()

"""build observables classes"""
phi_0 = phi_twin_track(FHM1, FHM2, psi1_t, psi2_t)
J1_0 = FHM1.phi_dependent_current(phi_0, psi1_t, lat)
J2_0 = FHM2.phi_dependent_current(phi_0, psi2_t, lat2)
if np.isclose(J1_0, J2_0):
    J_0 = J2_0
    print("Both systems give the same current.")
else:
    print("ERROR: initial current between the two systems after kick are not equal. Tracking failed.")
    sys.exit(1)

obs1 = observables(psi1_t, J_0, phi_0, FHM1)
obs2 = observables(psi2_t, J_0, phi_0, FHM2)
print("More classes built. This took {:.2f} seconds.".format(time() - ti))

"""evolve the states given the twinning hamiltonian"""
print("Starting to evolve the two systems by twinning Hamiltonian.")
ti = time()

for current_time in tqdm(t_p.times[:-1]):
    solver_args = dict(atol=1e-12)
    # print(psi_t.shape)
    psi1_t_new = evolve(v0=psi1_t, t0=current_time, times=np.array([current_time + t_p.delta]),
                        f=twinning_tracking_evolution, f_params=[FHM1, FHM2, lat, psi2_t.copy()], **solver_args)
    psi2_t_new = evolve(v0=psi2_t, t0=current_time, times=np.array([current_time + t_p.delta]),
                        f=twinning_tracking_evolution, f_params=[FHM1, FHM2, lat, psi1_t.copy()], **solver_args)

    psi1_t = np.squeeze(psi1_t_new)
    psi2_t = np.squeeze(psi2_t_new)
    phi = phi_twin_track(FHM1, FHM2, psi1_t, psi2_t)
    obs1.append_observables(psi1_t, phi)
    obs2.append_observables(psi2_t, phi)

print("Evolution complete. This took {:.2f} seconds".format(time() - ti))

"""make sure the specific system folder exists"""
try:
    os.mkdir("./Data/twin-{}U1-{}U2".format(param.U, param2.U))
except OSError as error:
    shutil.rmtree("./Data/twin-{}U1-{}U2".format(param.U, param2.U))
    os.mkdir("./Data/twin-{}U1-{}U2".format(param.U, param2.U))

"""save all observables to a file"""
outfile1 = './Data/twin-{}U1-{}U2/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}.npz'.\
    format(param.U, param2.U, param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc, param.a_scale
           , param.J_scale)

outfile2 = './Data/twin-{}U1-{}U2/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}.npz'.\
    format(param.U, param2.U, param2.L, param2.N_up, param2.N_down, param2.t0, param2.U, t_p.cycles, t_p.n_steps, param2.pbc,
           param2.a_scale, param2.J_scale)

expectations1 = dict()
expectations2 = dict()

obs1.save_observables(expectations1)
obs2.save_observables(expectations2)
print('Saving our expectations.')
ti = time()
np.savez(outfile1, **expectations1)
np.savez(outfile2, **expectations2)
print('Expectations saved. It took {:.2f} seconds.'.format(time() - ti))

print('Program finished. It took {:.2f} seconds to run'.format(time() - t_init))

plt.figure("currents and potentials")
plt.subplot(211)
plt.plot(t_p.times*lat.freq, expectations1["tracking_current"], label="System 1")
plt.plot(t_p.times*lat2.freq, expectations2["tracking_current"], "--", label="System 2")
plt.ylabel("$J(t)$")
plt.subplot(212)
plt.plot(t_p.times*lat.freq, expectations1["tracking_phi"], label="System 1")
plt.plot(t_p.times*lat2.freq, expectations2["tracking_phi"], "--", label="System 2")
plt.ylabel("$\\Phi(t)$")
plt.xlabel("Time (cycles)")
plt.legend()
plt.show()
