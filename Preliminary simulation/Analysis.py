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
param = unscaledparam(L=6, t0=0.52, U=0, pbc=True, field=32.9, F0=3, a=4)

"""generating our class of scaled parameters"""
"""this is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
cycles = 2 # time in cycles of field frequency
n_steps = int(2e3)
start = 0
stop = cycles
times, delta = np.linspace(start, stop, num=n_steps, endpoint=True, retstep=True)

"""prepare to load our data to be plotted"""
outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'.format(param.L, param.N_up,
                                                                                               param.N_down, param.t0,
                                                                                               param.U, cycles, n_steps,
                                                                                               param.pbc)
expectations = np.load(outfile)

"""plot out expectations"""
plt.figure("Current")
plt.xlabel("Time (cycles)")
plt.ylabel("$J(t)$")
plt.grid(True)
plt.tight_layout()
plt.plot(times, expectations['current'])
plt.show()