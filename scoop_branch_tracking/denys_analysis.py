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

########################################################################################################################

"""Generate our class for the unscaled parameters"""
"""these are primarily used for saving our data"""
param = unscaledparam(L=6, t0=0.52, U=0.5, pbc=True, field=32.9, F0=10, a=4, a_scale=1.0, J_scale=1, tracking=1)

"""generating our class of scaled parameters"""
"""this is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(2e4), plotting=1)

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
# data = data[41:20]
# data = [data[41], data[20]]
colouring = np.linspace(0, 1, len(data))
# plt.figure("control fields and currents")
# ax1 = plt.subplot(211)
# plt.plot(t_p.times, prelim_data['phi'])
fig, ax0 = plt.subplots(1,1)
for expect, color_code in zip(data, colouring):
    # print(np.allclose(expect['tracking_phi'].imag, 0))

    ax0.set(xlabel="Time (cycles)", ylabel="$\\Phi(t) - \\theta[\\psi(t)]$")
    # ax0.set_yticks([(3/2) * np.pi, np.pi, np.pi / 2, 0, - np.pi / 2, -np.pi, -(3/2) * np.pi])
    # ax0.set_yticklabels(["$\\frac{3\\pi}{2}$", "$\\pi$", "$\\frac{\\pi}{2}$", "$0$", "$-\\frac{\\pi}{2}$", "$-\\pi$", "$-\\frac{3\\pi}{2}$"])
    ax0.plot(
        t_p.times,
        expect['tracking_phi'][:len(t_p.times)].real  - np.angle(expect["tracking_neighbour"][:len(t_p.times)]),
        color=plt.cm.jet(color_code),
        alpha=0.5,
        linewidth=1,
    )
ax0.hlines(-np.pi/2, 0, 2 * np.pi, linestyles="dashed")
alpha = 0.5
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
for expect, color_code in zip(data, colouring):
    # print(np.allclose(expect['tracking_phi'].imag, 0))

    ax1.set(xlabel="Time (cycles)", ylabel="$\\Phi(t)$")
    ax1.set_yticks([(3/2) * np.pi, np.pi, np.pi / 2, 0, - np.pi / 2, -np.pi, -(3/2) * np.pi])
    ax1.set_yticklabels(["$\\frac{3\\pi}{2}$", "$\\pi$", "$\\frac{\\pi}{2}$", "$0$", "$-\\frac{\\pi}{2}$", "$-\\pi$", "$-\\frac{3\\pi}{2}$"])
    ax1.plot(
        t_p.times,
        expect['tracking_phi'][:len(t_p.times)].real,
        color=plt.cm.jet(color_code),
        alpha=alpha,
        linewidth=1,
    )
# plt.subplot(211)
# plt.plot(t_p.times, prelim_data['phi'])
# for expect, color_code in zip(data, colouring):
#     print(np.allclose(expect['tracking_phi'].imag, 0))
#
#     plt.ylabel("$\\Phi(t)$")
#
#     plt.plot(
#         t_p.times,
#         expect['tracking_phi'][:len(t_p.times)].real - np.angle(expect['tracking_neighbour'][:len(t_p.times)]),
#         color=plt.cm.jet(color_code),
#         alpha=0.3,
#         linewidth=3,
#     )
# plt.plot(t_p.times, prelim_data['current'].real)
for expect, color_code in zip(data, colouring):
    # print(np.allclose(expect['tracking_current'].imag, 0))

    ax2.set(xlabel="Time (cycles)", ylabel="$J(t)$")
    # ax2.set_yticks([-1, 0, 1])
    # ax2.set_yticklabels([-1, 0, 1])
    ax2.plot(
        t_p.times,
        expect['tracking_current'][:len(t_p.times)].real,
        color=plt.cm.jet(color_code),
        alpha=alpha,
        linewidth=1,
    )
# data = data[-15:]
# colouring = np.linspace(0, 1, len(data))
for expect, color_code in zip(data, colouring):
    # print(np.allclose(expect['tracking_current'].imag, 0))

    ax3.set(xlabel="Time (cycles)", ylabel="$H(t)$")
    # ax3.set_yticks([np.pi / 2, - np.pi / 2])
    # ax3.set_yticklabels(["$\\frac{\\pi}{2}$", "$-\\frac{\\pi}{2}$"])
    ax3.plot(
        t_p.times,
        expect['tracking_energy'][:len(t_p.times)].real,
        color=plt.cm.jet(color_code),
        alpha=alpha,
        linewidth=1,
    )

plt.figure("Fourier Transforms")
# phi = prelim_data['phi']
# N = len(phi)
# k = np.arange(N)
#
# # frequency range
# omegas = (k - N / 2) * np.pi / (0.5 * t_p.times[-1])
#
# # spectra of the
# spectrum = np.abs(
#     # used windows fourier transform to calculate the spectra
#     # rhttp://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
#     fftpack.fft((-1) ** k * signal.blackman(N) * phi)
# ) ** 2
# spectrum /= spectrum.max()
#
# # plt.subplot(211)
# plt.semilogy(omegas, spectrum)
# # plt.xlim(0,20)
# plt.show()

plt.subplot(212)

prev_max = 0
phi = prelim_data['current'][:len(t_p.times)].real
plt.ylabel("$\\mathcal{F}[J(t)]$")
method = 'welch'
min_spec = 8
max_harm = 60
gabor = 'fL'
w, spec = signal.welch(phi, 1/t_p.delta, nperseg=len(phi), scaling='spectrum')
# w *= 2. * np.pi / lat.field
plt.semilogy(
    w,
    spec
)
axes = plt.gca()
axes.set_xlim([0, max_harm])
if spec.max() > prev_max:
    prev_max = spec.max() * 5
axes.set_ylim([10 ** (-min_spec), prev_max])
xlines = [2 * i - 1 for i in range(1, 6)]
for expect, color_code in zip(data, colouring):
    # print(np.allclose(expect['tracking_phi'].imag, 0))
    phi = expect['tracking_current'][:len(t_p.times)].real
    plt.ylabel("$\\mathcal{F}[J(t)]$")
    w, spec = signal.welch(phi, 1/t_p.delta, nperseg=len(phi), scaling='spectrum')
    # w *= 2. * np.pi / lat.field
    plt.semilogy(
        w,
        spec,
        color=plt.cm.jet(color_code),
        alpha=alpha,
        linewidth=3,
    )
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    if spec.max() > prev_max:
        prev_max = spec.max() * 5
    axes.set_ylim([10 ** (-min_spec), prev_max])
    # xlines = [2 * i - 1 for i in range(1, 6)]
    # plt.plot(
    #     t_p.times,
    #     fft(phi),
    #     color=plt.cm.jet(color_code),
    #     alpha=0.3,
    #     linewidth=3,
    # )

# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('Harmonic Order')
# plt.ylabel('HHG spectra of $J(t)$')
# plt.legend(loc='upper right')

plt.subplot(211)

prev_max = 0
    # print(np.allclose(expect['tracking_phi'].imag, 0))
phi = prelim_data['phi'][:len(t_p.times)].real
plt.ylabel("$\\mathcal{F}[\\Phi(t)]$")
method = 'welch'
min_spec = 8
max_harm = 30
gabor = 'fL'
w, spec = signal.welch(phi, 1/t_p.delta, nperseg=len(phi), scaling='spectrum')
# w *= 2. * np.pi / lat.field
plt.semilogy(
    w,
    spec,
)
axes = plt.gca()
axes.set_xlim([0, max_harm])
if spec.max() > prev_max:
    prev_max = spec.max() * 5
axes.set_ylim([10 ** (-min_spec), prev_max])
xlines = [2 * i - 1 for i in range(1, 6)]

for expect, color_code in zip(data, colouring):
    # print(np.allclose(expect['tracking_phi'].imag, 0))
    phi = expect['tracking_phi'][:len(t_p.times)].real
    plt.ylabel("$\\mathcal{F}[\\Phi(t)]$")
    w, spec = signal.welch(phi, 1/t_p.delta, nperseg=len(phi), scaling='spectrum')
    # w *= 2. * np.pi / lat.field
    plt.semilogy(
        w,
        spec,
        color=plt.cm.jet(color_code),
        alpha=alpha,
        linewidth=3,
    )
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    if spec.max() > prev_max:
        prev_max = spec.max() * 5
    axes.set_ylim([10 ** (-min_spec), prev_max])
    xlines = [2 * i - 1 for i in range(1, 6)]
    # plt.plot(
    #     t_p.times,
    #     fft(phi),
    #     color=plt.cm.jet(color_code),
    #     alpha=0.3,
    #     linewidth=3,
    # )

for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('Harmonic Order')
# plt.ylabel('HHG spectra of $\\Phi(t)$')
# plt.legend(loc='upper right')
plt.show()

# plt.figure("Spectra Difference")
# expect1 = data[0]
# expect2 = data[1]
# phi1 = expect1['tracking_phi'][:len(t_p.times)].real
# phi2 = expect2['tracking_phi'][:len(t_p.times)].real
# plt.ylabel("$\\mathcal{F}[\\Phi(t)]$")
# w1, spec1 = signal.welch(phi1, 1/t_p.delta, nperseg=len(phi1), scaling='spectrum')
# w2, spec2 = signal.welch(phi2, 1/t_p.delta, nperseg=len(phi1), scaling='spectrum')
# # w *= 2. * np.pi / lat.field
# plt.semilogy(
#     w1,
#     np.abs(spec1 - spec2),
#     alpha=alpha,
#     linewidth=3,
# )
# axes = plt.gca()
# axes.set_xlim([0, max_harm])
# if spec.max() > prev_max:
#     prev_max = spec.max() * 5
# axes.set_ylim([10 ** (-min_spec), prev_max])
# xlines = [2 * i - 1 for i in range(1, 6)]
#     # plt.plot(
#     #     t_p.times,
#     #     fft(phi),
#     #     color=plt.cm.jet(color_code),
#     #     alpha=0.3,
#     #     linewidth=3,
#     # )
#
# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
# plt.xlabel('Harmonic Order')
# # plt.ylabel('HHG spectra of $\\Phi(t)$')
# # plt.legend(loc='upper right')
# plt.show()

