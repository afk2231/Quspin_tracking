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
from scipy import signal

params = {
    'axes.labelsize': 42,
    # 'legend.fontsize': 28,
    'legend.fontsize': 25,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    # 'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True,
    # 'figure.figsize': (12, 16),
    'figure.figsize': (16, 12),
    'lines.linewidth' : 3,
    'lines.markersize' : 15
}
plt.rcParams.update(params)

delta_u_grid = [0.1, 0.5, 1.0]

"""Generate our class for the unscaled parameters"""
param1 = unscaledparam(L=6, t0=0.52, U=5, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

"""generating our class of scaled parameters"""
lat1 = hhg(field=param1.field, nup=param1.N_up, ndown=param1.N_down, nx=param1.L, ny=0, U=param1.U, t=param1.t0, F0=param1.F0
          , a=param1.a, pbc=param1.pbc)

"""Generate our class for the unscaled parameters"""
param2 = []
for delta_u in delta_u_grid:
    param2.append(unscaledparam(L=6, t0=0.52, U=param1.U/param1.t0 + delta_u, pbc=True, field=32.9, F0=10, a=4, a_scale=1,
                                J_scale=1, tracking=1))
"""generating our class of scaled parameters"""
lat2 = []
for param in param2:
    lat2.append(hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U,
                    t=param.t0, F0=param.F0, a=param.a, pbc=param.pbc))

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat1, cycles=10, nsteps=int(2e3), plotting=1)

"""prepare to load our data to be plotted"""
outfile1 = './Data/twin-{}U1-{}U2/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}.npz'.format(
    param1.U, param2[0].U, param1.L, param1.N_up, param1.N_down, param1.t0, param1.U, t_p.cycles, t_p.n_steps, param1.pbc,
    param1.a_scale, param1.J_scale)
outfile2 = []
for param in param2:
    outfile2.append('./Data/twin-{}U1-{}U2/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}.npz'.format(
    param1.U, param.U, param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc,
    param.a_scale, param.J_scale))

expectations1 = np.load(outfile1)
expectations2 = []
for outfile in outfile2:
    expectations2.append(np.load(outfile))

"""plot out expectations"""

"""Plotting field"""
plt.figure("Control field and Current")
plt.subplot(211)
# plt.xlabel("Time (cycles)")
plt.ylabel("$\\mathcal{F}(\\Phi(t))$")
# plt.grid(True)
plt.tight_layout()
prev_max = 0
method = 'welch'
min_spec = 8
max_harm = 40
gabor = 'fL'
for expectation, delta_u in zip(expectations2, delta_u_grid):
    phi = expectation["tracking_phi"].real
    w, spec = signal.welch(phi, 1 / t_p.delta, nperseg=len(phi), scaling='spectrum')
    # w *= 2. * np.pi / lat.field
    plt.semilogy(
        w,
        spec,
        linewidth=2,
        label="$\\Delta U / t_0 = ${:.2f}, $U_0 / t_0 = ${:.2f}".format(delta_u, param1.U / param1.t0)
    )
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    if spec.max() > prev_max:
        prev_max = spec.max() * 5
    axes.set_ylim([10 ** (-min_spec), prev_max])
plt.legend()

"""Plotting current"""
# plt.figure("Current")
plt.subplot(212)
plt.xlabel("Frequency")
plt.ylabel("$\\mathcal{F}(J(t))$")
# plt.grid(True)
plt.tight_layout()
max_harm = 40
for expectation, delta_u in zip(expectations2, delta_u_grid):
    current = expectation["tracking_current"].real
    w, spec = signal.welch(current, 1 / t_p.delta, nperseg=len(current), scaling='spectrum')
    # w *= 2. * np.pi / lat.field
    plt.semilogy(
        w,
        spec,
        linewidth=2,
        label="$\\Delta U / t_0 = ${:.2f}, $U_0 / t_0 = ${:.2f}".format(delta_u, param1.U / param1.t0)
    )
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    if spec.max() > prev_max:
        prev_max = spec.max() * 5
    axes.set_ylim([10 ** (-min_spec), prev_max])
plt.legend()
plt.savefig('./Fig5.pdf', bbox_inches='tight')
plt.show()
