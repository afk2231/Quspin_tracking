##################################################################
#   Analysis for data generated by various H-R tracking methods
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
param = unscaledparam(L=6, t0=0.52, U=0.5, pbc=True, field=32.9, F0=10, a=4, a_scale=1, J_scale=1, tracking=1)

"""generating our class of scaled parameters"""
"""this is used for most of the calculations"""
lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0, F0=param.F0
          , a=param.a, pbc=param.pbc)

"""setup our evolution time parameters"""
t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(1e4), plotting=1)

"""prepare to load our data to be plotted"""
original = True
if original:
    outfile = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}' \
              '-method={}.npz'.format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps,
                                      param.pbc, param.a_scale, param.J_scale, 'R_tracking')
    expectations = np.load(outfile)
else:

    outfile2 = './Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc:a_scale={:.2f}-J_scale={:.2f}' \
          '-method={}.npz'.format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps,
                                  param.pbc, param.a_scale, param.J_scale, 'delay_R_tracking')

    expectations2 = np.load(outfile2)

"""plot out expectations"""

if original:
    """Plotting field"""
    plt.figure("Control field")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\Phi(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, (expectations['phi'] - np.angle(expectations['hop_left_op']))/(np.pi/2))
    plt.plot(t_p.times, (expectations['tracking_phi_R_tracking'] - np.angle(expectations['tracking_neighbour_R_tracking']))/(np.pi/2), ".")
    plt.show()

    """Plotting current"""
    plt.figure("Current")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$J(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, expectations['current'])
    plt.plot(t_p.times, expectations['tracking_current_R_tracking'])
    plt.show()

    """Plotting energy"""
    plt.figure("energy")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$E(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, expectations['H'])
    plt.plot(t_p.times, expectations['tracking_energy_R_tracking'], ".")
    plt.show()

    # """Plotting R"""
    # plt.figure("R")
    # plt.xlabel("Time (cycles)")
    # plt.ylabel("$R(t)$")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.plot(t_p.times, np.abs(expectations['hop_left_op']))
    # plt.plot(t_p.times, np.abs(expectations['tracking_neighbour_R_tracking']), ".")
    # plt.show()

    """Plotting theta"""
    plt.figure("angle")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\theta(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, np.angle(expectations['hop_left_op']))
    plt.plot(t_p.times, np.angle(expectations['tracking_neighbour_R_tracking']), ".")
    plt.show()

    # """Plotting theta"""
    # plt.figure("real")
    # plt.xlabel("Time (cycles)")
    # plt.ylabel("$NN_r(t)$")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.plot(t_p.times, expectations['hop_left_op'].real)
    # plt.plot(t_p.times, expectations['tracking_neighbour_R_tracking'].real, ".")
    # plt.show()

    """Plotting number of particles"""
    plt.figure("N")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\frac{d \\langle N \\rangle}{dt}$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, np.gradient(expectations['tracking_pnumber_R_tracking'], t_p.delta))
    plt.show()

    plt.figure("Nj")
    for j in range(lat.nx - 1):
        plt.subplot(int(f"{lat.nx}{1}{j + 1}"))
        plt.ylabel("$\\langle N \\rangle$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, expectations["tracking_pnumbersite" + str(j) + "_R_tracking"])
    if lat.pbc:
        plt.subplot(int(f"{lat.nx}{1}{lat.nx}"))
        plt.xlabel("Time (cycles)")
        plt.ylabel("$\\langle N \\rangle$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, expectations["tracking_pnumbersite" + str(lat.nx - 1) + "_R_tracking"])

    # plt.figure("current at site 0")
    # plt.xlabel("Time (cycles)")
    # plt.ylabel("$\\frac{d \\langle N \\rangle}{dt}$")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.plot(t_p.times, expectations['tracking_current_R_tracking']/6)
    # plt.plot(t_p.times, expectations['tracking_pcurrentsite0_R_tracking'])

    """Continuity checks"""
    plt.figure("Continuity Check")
    if lat.pbc:
        plt.subplot(int(f"{lat.nx}{1}{1}"))
        plt.xlabel("Time (cycles)")
        plt.ylabel("$\\frac{d \\langle N \\rangle}{dt}$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, np.gradient(expectations["tracking_pnumbersite" + str(0) + "_R_tracking"], t_p.delta / lat.freq))
        plt.plot(t_p.times, (expectations['tracking_pcurrentsite' + str(0) + '_R_tracking']
                             - expectations['tracking_pcurrentsite' + str(lat.nx - 1) + '_R_tracking']) / lat.a)

    for _ in range(1,lat.nx):
        plt.subplot(int(f"{lat.nx}{1}{_ + 1}"))
        plt.ylabel("$\\frac{d \\langle N \\rangle}{d t}$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, np.gradient(expectations["tracking_pnumbersite" + str(_) + "_R_tracking"], t_p.delta/ lat.freq))
        plt.plot(t_p.times, (expectations['tracking_pcurrentsite' + str(_) + "_R_tracking"]
                             - expectations['tracking_pcurrentsite' + str(_ - 1) + "_R_tracking"])/lat.a)

    plt.figure("Ehrenfest Theorem")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\frac{d J}{dt}$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, np.gradient(expectations['tracking_current_R_tracking'], t_p.delta / lat.freq))
    D = expectations['tracking_neighbour_R_tracking']
    phi = expectations['tracking_phi_R_tracking']
    ehrenfest2 = -2 * lat.a * lat.t * (np.gradient(np.abs(D), t_p.delta / lat.freq) * np.sin(phi - np.angle(D))
                                       + np.abs(D) * np.cos(phi - np.angle(D))
                                       * np.gradient(phi - np.angle(D), t_p.delta / lat.freq))
    plt.plot(t_p.times, ehrenfest2)
else:
    """Plotting field"""
    plt.figure("Control field")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\Phi(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, (expectations2['phi'] - np.angle(expectations2['hop_left_op']))/(np.pi/2))
    plt.plot(t_p.times, (expectations2['tracking_phi_delay_R_tracking'] - np.angle(expectations2['tracking_neighbour_delay_R_tracking']))/(np.pi/2), ".")
    plt.show()

    """Plotting current"""
    plt.figure("Current")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$J(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, expectations2['current'])
    plt.plot(t_p.times, expectations2['tracking_current_delay_R_tracking'])
    plt.show()

    """Plotting energy"""
    plt.figure("energy")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$E(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, expectations2['H'])
    plt.plot(t_p.times, expectations2['tracking_energy_delay_R_tracking'], ".")
    plt.show()

    # """Plotting R"""
    # plt.figure("R")
    # plt.xlabel("Time (cycles)")
    # plt.ylabel("$R(t)$")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.plot(t_p.times, np.abs(expectations['hop_left_op']))
    # plt.plot(t_p.times, np.abs(expectations['tracking_neighbour_R_tracking']), ".")
    # plt.show()

    """Plotting theta"""
    plt.figure("angle")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\theta(t)$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, np.angle(expectations2['hop_left_op']))
    plt.plot(t_p.times, np.angle(expectations2['tracking_neighbour_delay_R_tracking']), ".")
    plt.show()

    # """Plotting theta"""
    # plt.figure("real")
    # plt.xlabel("Time (cycles)")
    # plt.ylabel("$NN_r(t)$")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.plot(t_p.times, expectations['hop_left_op'].real)
    # plt.plot(t_p.times, expectations['tracking_neighbour_R_tracking'].real, ".")
    # plt.show()

    """Plotting number of particles"""
    plt.figure("N")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\frac{d \\langle N \\rangle}{dt}$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, np.gradient(expectations2['tracking_pnumber_delay_R_tracking'], t_p.delta))
    plt.show()

    plt.figure("Nj")
    for j in range(lat.nx - 1):
        plt.subplot(int(f"{lat.nx}{1}{j + 1}"))
        plt.ylabel("$\\langle N \\rangle$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, expectations2["tracking_pnumbersite" + str(j) + "_delay_R_tracking"])
    if lat.pbc:
        plt.subplot(int(f"{lat.nx}{1}{lat.nx}"))
        plt.xlabel("Time (cycles)")
        plt.ylabel("$\\langle N \\rangle$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, expectations2["tracking_pnumbersite" + str(lat.nx - 1) + "_delay_R_tracking"])

    # plt.figure("current at site 0")
    # plt.xlabel("Time (cycles)")
    # plt.ylabel("$\\frac{d \\langle N \\rangle}{dt}$")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.plot(t_p.times, expectations['tracking_current_R_tracking']/6)
    # plt.plot(t_p.times, expectations['tracking_pcurrentsite0_R_tracking'])

    """Continuity checks"""
    plt.figure("Continuity Check")
    if lat.pbc:
        plt.subplot(int(f"{lat.nx}{1}{1}"))
        plt.xlabel("Time (cycles)")
        plt.ylabel("$\\frac{d \\langle N \\rangle}{dt}$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, np.gradient(expectations2["tracking_pnumbersite" + str(0) + "_delay_R_tracking"], t_p.delta / lat.freq))
        plt.plot(t_p.times, (expectations2['tracking_pcurrentsite' + str(0) + '_delay_R_tracking']
                             - expectations2['tracking_pcurrentsite' + str(lat.nx - 1) + '_delay_R_tracking']) / lat.a)

    for _ in range(1,lat.nx):
        plt.subplot(int(f"{lat.nx}{1}{_ + 1}"))
        plt.ylabel("$\\frac{d \\langle N \\rangle}{d t}$")
        plt.grid(True)
        plt.tight_layout()
        plt.plot(t_p.times, np.gradient(expectations2["tracking_pnumbersite" + str(_) + "_delay_R_tracking"], t_p.delta/ lat.freq))
        plt.plot(t_p.times, (expectations2['tracking_pcurrentsite' + str(_) + "_delay_R_tracking"]
                             - expectations2['tracking_pcurrentsite' + str(_ - 1) + "_delay_R_tracking"])/lat.a)

    plt.figure("Ehrenfest Theorem")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\\frac{d J}{dt}$")
    plt.grid(True)
    plt.tight_layout()
    plt.plot(t_p.times, np.gradient(expectations2['tracking_current_delay_R_tracking'], t_p.delta / lat.freq))
    D = expectations2['tracking_neighbour_delay_R_tracking']
    phi = expectations2['tracking_phi_delay_R_tracking']
    ehrenfest2 = -2 * lat.a * lat.t * (np.gradient(np.abs(D), t_p.delta / lat.freq) * np.sin(phi - np.angle(D))
                                       + np.abs(D) * np.cos(phi - np.angle(D))
                                       * np.gradient(phi - np.angle(D), t_p.delta / lat.freq))
    plt.plot(t_p.times, ehrenfest2)
