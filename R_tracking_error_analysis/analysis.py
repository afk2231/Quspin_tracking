##################################################################
#   Analysis for data generated by this preliminary simulation
#   Usually only used for bug fixing the prelim sim.
##################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
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
param = unscaledparam(L=6, t0=0.52, U=0.5, pbc=True, field=32.9, F0=10, a=4)


"""prepare to load our data to be plotted"""
outfile = './Data/errors:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}pbc.npz'.format(param.L, param.N_up, param.N_down,
                                                                                 param.t0, param.U, 2, param.pbc)
errors = np.load(outfile)

steps = np.linspace(1000, 10000, len(errors['tar_err']))

"""plot out expectations"""

plt.figure("absolute error of untracked simulation")
plt.title("Absolute time-averaged error of untracked simulation")
plt.xlabel("Number of Steps")
plt.ylabel("$\\langle \\epsilon_{abs} \\rangle_t$")
plt.grid(True)
plt.tight_layout()
plt.loglog(steps, errors['taa_err'])
plt.show()

plt.figure("absolute error of R-tracked simulation")
plt.title("Absolute time-averaged error of R-tracked simulation")
plt.xlabel("ln(Number of Steps)")
plt.ylabel("$\\ln(\\langle \\epsilon_{abs} \\rangle_t)$")
plt.grid(True)
plt.tight_layout()
plt.loglog(steps, errors['taa_err_R'])
plt.show()

plt.figure("relative error")
plt.title("Relative time-averaged error")
plt.xlabel("Number of Steps")
plt.ylabel("$\\langle \\epsilon_{rel} \\rangle_t$")
plt.grid(True)
plt.tight_layout()
plt.loglog(steps, errors['tar_err'])
plt.show()