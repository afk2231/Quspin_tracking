import numpy as np
import matplotlib.pyplot as plt
import glob
from classes.time_param.t_param import time_evolution_params
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.perimeter_params.tools import parameter_instantiate as hhg
import sys
from scipy import signal
from scipy import fftpack
from matplotlib.colors import SymLogNorm
sys.path.append('../')

pltparams = {
    'axes.labelsize': 24,
    # 'legend.fontsize': 28,
    'legend.fontsize': 12,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.figsize': [5.2 * 3.375, 3.5 * 3.375],
    'text.usetex': True
}
plt.rcParams.update(pltparams)
u_grid = [0.0, 0.2, 0.3]

########################################################################################################################

"""Generate our class for the unscaled parameters"""
"""these are primarily used for saving our data"""
params = []
for u in u_grid:
    params.append(
        unscaledparam(L=6, t0=0.52, U=u, pbc=True, field=32.9, F0=10, a=4, a_scale=1.0, J_scale=1, tracking=1)
    )

"""generating our class of scaled parameters"""
"""this is used for most of the calculations"""
lats = []
for param in params:
    lats.append(
        hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
              , a=param.a, pbc=param.pbc)
    )

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lats[0], cycles=2, nsteps=int(2e4), plotting=1)
folder_names = []
for param in params:
    folder_names.append(
        './Data/{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbcF0={:.2f}-' \
                        'J_scale={:.2f}'.format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles,
                                                t_p.n_steps, param.pbc, param.F0, param.J_scale)
    )

prelim_datas = []
for param in params:
    prelim_file = '../Preliminary simulation/Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'\
        .format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc)
    prelim_datas.append(
        np.load(prelim_file)
    )

########################################################################################################################
file_ns = []
for folder_name in folder_names:
    file_ns.append(
        glob.glob(folder_name + "/*.npz")
    )

datas = []
for file_names in file_ns:
    data = [np.load(fname) for fname in file_names]
    datas.append(
        data
    )
# order the data
data_select = [None, -1, -1]
for j in range(3):
    data = datas[j]
    prelim_data = prelim_datas[j]
    data_s = data_select[j]

    data.sort(
        # key=lambda expect: np.abs(expect['tracking_phi'].real[-1])
        key=lambda expect: np.linalg.norm(prelim_data['current'] - expect['tracking_current'][:len(prelim_data['current'])])
    )
    datas[j] = data[:data_s]

    data.sort(
        # key=lambda expect: expect['tracking_phi'].real[-1]
        key=lambda expect: np.abs(expect['tracking_phi'].real[-1])
        # key=lambda expect: np.linalg.norm(prelim_data['current'] - expect['tracking_current'][:len(prelim_data['current'])])
    )

# print(datas[0])
# sys.exit(1)
colourings = []
for data in datas:
    colourings.append(
        np.linspace(0, 1, len(data))
    )
# plt.figure("control fields and currents")
# ax1 = plt.subplot(211)
# plt.plot(t_p.times, prelim_data['phi'])
fig, ax = plt.subplots(1, 3)
ax[0].set(ylabel="$\\langle \hat{H}\\rangle(t)$")
for data, colouring, ax_n in zip(datas, colourings, ax):
    for expect, color_code in zip(data, colouring):
        # print(np.allclose(expect['tracking_phi'].imag, 0))

        ax_n.set(xlabel="Time (cycles)")
        # ax0.set_yticks([(3/2) * np.pi, np.pi, np.pi / 2, 0, - np.pi / 2, -np.pi, -(3/2) * np.pi])
        # ax0.set_yticklabels(["$\\frac{3\\pi}{2}$", "$\\pi$", "$\\frac{\\pi}{2}$", "$0$", "$-\\frac{\\pi}{2}$", "$-\\pi$", "$-\\frac{3\\pi}{2}$"])
        ax_n.plot(
            t_p.times,
            expect['tracking_energy'][:len(t_p.times)].real,
            color=plt.cm.jet(color_code),
            alpha=0.5,
            linewidth=1,
        )
plt.show()
