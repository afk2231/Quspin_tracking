##################################################################
#   Analysis for data generated by this preliminary simulation
#   Usually only used for bug fixing the prelim sim.
##################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from classes.perimeter_params.tools import parameter_instantiate as hhg
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params

pltparams = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [5.2 * 3.375, 3.5 * 3.375],
    'text.usetex': True
}
plt.rcParams.update(pltparams)

"""Generate our class for the unscaled parameters"""
"""these are primarily used for saving our data"""
param = unscaledparam(L=6, t0=0.52, U=1, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

"""generating our class of scaled parameters"""
"""this is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(2e3), plotting=1)

"""prepare to load our data to be plotted"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}.npz'.\
    format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc, param.a_scale
           , param.J_scale)
expectations = np.load(outfile)

"""plot out expectations"""

"""Plotting field"""
plt.figure("Control field")
plt.xlabel("Time (cycles)")
plt.ylabel("$\\Phi(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(t_p.times, expectations['phi'])
plt.plot(t_p.times, expectations['tracking_phi'], ".")
plt.show()

"""Plotting current"""
plt.figure("Current")
plt.xlabel("Time (cycles)")
plt.ylabel("$J(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(t_p.times, expectations['current'])
plt.plot(t_p.times, expectations['tracking_current'], ".")
plt.show()

"""Plotting energy"""
plt.figure("energy")
plt.xlabel("Time (cycles)")
plt.ylabel("$E(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(t_p.times, expectations['H'])
plt.plot(t_p.times, expectations['tracking_energy'], ".")
plt.show()

"""Plotting R"""
plt.figure("R")
plt.xlabel("Time (cycles)")
plt.ylabel("$R(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(t_p.times, np.abs(expectations['hop_left_op']))
plt.plot(t_p.times, np.abs(expectations['tracking_neighbour']), ".")
plt.show()

"""Plotting theta"""
plt.figure("angle")
plt.xlabel("Time (cycles)")
plt.ylabel("$\\theta(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(t_p.times, np.angle(expectations['hop_left_op']))
plt.plot(t_p.times, np.angle(expectations['tracking_neighbour']), ".")
plt.show()

"""Plotting theta"""
plt.figure("real")
plt.xlabel("Time (cycles)")
plt.ylabel("$NN_r(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(t_p.times, expectations['hop_left_op'].real)
plt.plot(t_p.times, expectations['tracking_neighbour'].real, ".")
plt.show()