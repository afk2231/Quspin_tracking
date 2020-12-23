#######################################################################################################################
#   Basic simulation used to generate data used generate state for H-tracking.
#
#######################################################################################################################

from __future__ import print_function, division
import os
import sys
from quspin.tools.measurements import obs_vs_time
from classes.fermi_hubbard_class.setup_fermi_hubbard import Fermi_Hubbard
from classes.perimeter_params.tools import parameter_instantiate as hhg  # Used for scaling units.
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params
from functions.original_evolution_equations.evol_eqs import phi
import psutil
import numpy as np  # general math functions
from time import time  # tool for calculating computation time

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
param = unscaledparam(L=6, t0=0.52, U=0.6, pbc=False, field=32.9, F0=10, a=4)

"""Generating our class of scaled parameters. This is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(4e3))

"""setup quspin operators and lists"""
FHM = Fermi_Hubbard(lat, t_p.cycles)

"""get inital energy and state"""
ti = time()
E, psi_0 = FHM.operator_dict['H'].eigsh(k=1, which='SA')
print("Initial state and energy calculated. It took {:.2f} seconds to calculate".format(time() - ti))
print("Ground state energy was calculated to be {:.2f}".format(E[0]))

"""evolve that energy and state"""
print('Starting Evolution')
ti = time()
psi_t = FHM.operator_dict["H"].evolve(psi_0, 0.0, t_p.times)
psi_t = np.squeeze(psi_t)
print('Evolution finished. It took {:.2f} seconds to evolve the system'.format(time() - ti))


"""save all observables to a file"""
outfile = './Data/state/state1.npz'

state = dict(psi_0=psi_t[:, -1])

print(FHM.operator_dict['current'].expt_value(state['psi_0']))
print('Saving our state.')
ti = time()
np.savez(outfile, **state)
print('State saved. It took {:.2f} seconds.'.format(time() - ti))

print('Program finished. It took {:.2f} seconds to run'.format(time() - t_init))