#######################################################################################################################
#   Error analysis simulation used to find the optimal step size for simulation. Will use a case where the tracking
#   will not fail to reproduce original control field. Pulls data from prelim, but also determines the error from
#   the preliminary simulation. The error produced here is the time averaged error.
#
#   This code can repurposed to look at time vs stepsize as well, but I don't feel like I need to implement it now.
#######################################################################################################################
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
from functions.H_R_tracking_equations.H_R_evol_eqs import R_tracking_evolution_equation
from functions.original_evolution_equations.evol_eqs import phi as phi_bm
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


"""pull data from an acceptable preliminary simulation"""
### MAKE SURE THAT THE DATA PULLED FROM THIS HAS A LOW F0, OR ELSE THE ERROR ANALYSIS MAY GO CRAZY ###
acceptable_data = \
    '../Preliminary simulation/Data/expectations:6sites-3up-3down-0.52t0-0.26U-2cycles-10000steps-Truepbc.npz'
a_d = dict(np.load(acceptable_data))

"""build everything needed for both simulations"""

# generating parameter classes
param = unscaledparam(L=6, t0=0.52, U=0.5, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

acceptable_current = UnivariateSpline(np.linspace(0, 2/lat.freq, num=len(a_d['current'])), a_d['current'], s=0)


param_track = unscaledparam(L=6, t0=0.52, U=0.5, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

lat_track = hhg(field=param_track.field, nup=param_track.N_up, ndown=param_track.N_down, nx=param_track.L, ny=0,
                U=param_track.U, t=param_track.t0, F0=param_track.F0, a=param_track.a, pbc=param_track.pbc)

# creating fermihubbard classes
FHM = Fermi_Hubbard(lat, 2)

FHM_track = Fermi_Hubbard(lat_track, 2)

FHM_track.create_commutator()

# generating groundstates
E, psi_0 = FHM.operator_dict['H'].eigsh(k=1, which='SA')

E_track, psi_0_track = FHM.operator_dict['H'].eigsh(k=1, which='SA')


"""start running R tracking and prelim simulation starting with 1000 steps incrimenting by 100 until 10000"""
t_a_absolute_err = []
t_a_absolute_err_R_track = []
t_a_relative_err = []
min_steps = 1000
max_steps = 10000
delta_steps = 100
for j in range(int((max_steps - min_steps)/delta_steps) + 1):
    # create our time parameters class
    n_steps = min_steps + j * delta_steps
    t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=n_steps)

    # evolve the untracked state
    psi_t = FHM.operator_dict["H"].evolve(psi_0, 0.0, t_p.times)
    psi_t = np.squeeze(psi_t)

    # find expectation from these states
    current = obs_vs_time(psi_t, t_p.times, dict(J=FHM.operator_dict['current']))

    # define the tracked time parameters class and the target J
    t_p_track = time_evolution_params(perimeter_params=lat_track, cycles=2, nsteps=n_steps)
    J_target = UnivariateSpline(t_p_track.times, param_track.J_scale * current['J'], s=0)

    # generate our extended vector and create our observables class for things to work (we don't actually use it for its
    #   intended purpose here)
    gamma_t = np.append(np.squeeze(psi_0), 0.0)

    obs = observables(gamma_t[:-1], J_target(0.0), phi_J_track(lat_track, 0.0, J_target, FHM_track, gamma_t[:-1]),
                      FHM_track)
    R_current = [J_target(0.0), ]

    # generate the current through runga-kutta of the state
    for newtime in tqdm(t_p.times[:-1]):
        solver_args = dict(atol=1e-4)
        gamma_t = evolve(v0=gamma_t, t0=newtime, times=np.array([newtime + t_p.delta]), f=R_tracking_evolution_equation,
                         f_params=[FHM_track, obs, J_target], solver_name='dopri5', **solver_args)
        gamma_t = gamma_t.reshape(-1)
        D = FHM.operator_dict['hop_left_op'].expt_value(gamma_t[:-1])
        phi_t = obs.phi_init + np.angle(D) - gamma_t[-1]
        R_current.append(
            -1j * lat_track.a * lat_track.t * (np.exp(-1j * phi_t) * D - np.exp(1j * phi_t) * D.conj())
        )

    # generate our time averaged errors through our observables.
    t_a_absolute_err.append(np.sum(np.abs(acceptable_current(t_p.times) - current['J']))/n_steps)
    t_a_absolute_err_R_track.append(np.sum(np.abs(acceptable_current(t_p_track.times) - R_current))/n_steps)
    t_a_relative_err.append(np.sum(np.abs(current['J'] - R_current))/n_steps)

"""save the errors"""

outfile = './Data/errors:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}pbc.npz'.format(param.L, param.N_up, param.N_down,
                                                                                 param.t0, param.U, 2,
                                                                                 param.pbc)

error = dict(taa_err=t_a_absolute_err, taa_err_R=t_a_absolute_err_R_track, tar_err=t_a_relative_err)

np.savez(outfile, **error)

