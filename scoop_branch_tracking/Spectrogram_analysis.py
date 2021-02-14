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
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [5.2 * 3.375, 3.5 * 3.375],
    'text.usetex': True
}
plt.rcParams.update(pltparams)

########################################################################################################################

"""Generate our class for the unscaled parameters"""
"""these are primarily used for saving our data"""
param = unscaledparam(L=6, t0=0.52, U=0, pbc=True, field=32.9, F0=10, a=4, a_scale=1.0, J_scale=1, tracking=1)

"""generating our class of scaled parameters"""
"""this is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=5, nsteps=int(4e4), plotting=1)

folder_name = './Data/{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbcF0={:.2f}-' \
                        'J_scale={:.2f}'.format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles,
                                                t_p.n_steps, param.pbc, param.F0, param.J_scale)

prelim_file = '../Preliminary simulation/Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz'\
    .format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc)
prelim_data = np.load(prelim_file)

########################################################################################################################
file_names = glob.glob(folder_name + "/*.npz")

data = [np.load(fname) for fname in file_names]
# order the data
data.sort(
    # key=lambda expect: np.abs(expect['tracking_phi'].real[-1])
    key=lambda expect: np.linalg.norm(prelim_data['current'] - expect['tracking_current'][:len(prelim_data['current'])])
)
data = data[:]
data.sort(
    # key=lambda expect: expect['tracking_phi'].real[-1]
    key=lambda expect: np.abs(expect['tracking_phi'].real[-1])
    # key=lambda expect: np.linalg.norm(prelim_data['current'] - expect['tracking_current'][:len(prelim_data['current'])])
)
data = [data[41], data[20]]
colouring = np.linspace(0, 1, len(data))
plt.figure("Phi spectrograms")
phi = prelim_data['phi'][:len(t_p.times)].real
fs = len(phi)/(10 / lat.freq)
f, t, Sxx = signal.spectrogram(phi, fs, nperseg=10)
plt.pcolormesh(t, f, Sxx/Sxx.max(), norm=SymLogNorm(vmin=1e-6, vmax=1., linthresh=1e-15))
plt.colorbar()
plt.ylim(0, 5)
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.show()

for expect, color_code in zip(data, colouring):
    plt.figure("Phi spectrograms" + str(color_code))
    phi = expect['tracking_phi'][:len(t_p.times)].real
    fs = len(phi)/(10 / lat.freq)
    f, t, Sxx = signal.spectrogram(phi, fs)
    plt.pcolormesh(t, f, Sxx/Sxx.max(), norm=SymLogNorm(vmin=1e-6, vmax=1., linthresh=1e-15))
    plt.colorbar()
    plt.ylim(0, 5)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.show()