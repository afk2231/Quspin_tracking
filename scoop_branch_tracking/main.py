from __future__ import print_function, division
import os
import shutil
import copy
import sys
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
from classes.fermi_hubbard_class.setup_fermi_hubbard import Fermi_Hubbard
from classes.perimeter_params.tools import parameter_instantiate as hhg  # Used for scaling units.
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params
from classes.observables_class.observables import observables
from functions.original_tracking_equations.original_tracking_equations import phi_J_track_with_branches, phi_J_track
from functions.original_tracking_equations.original_tracking_equations import original_tracking_evolution_with_branches
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import psutil
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from matplotlib import pyplot as plt
from tqdm import tqdm
from scoop import futures
PI = np.pi
threads = 2
os.environ['OMP_NUM_THREADS'] = '{}'.format(threads)
os.environ['MKL_NUM_THREADS'] = '{}'.format(threads)
"""
Program produces all expectations related to all branch points associated with a single target current function.
"""

def schrod_evol(start_time, psi_0, t_p, fhm, j_target, param, l, obs, branch_num, branch_string, split_check):
    """
    Handles the Schrodinger equation evolution. Built in recursion for handling all branches that the function will
    hit. When the control field is near an odd multiple of half pi: (2k + 1)*(pi/2), scoop will spawn a new worker to
    continue the evolution on that branch.

    This was built to use quspin's architecture for the FHM wavefunction and operators, it was also build for scoop's
    built in parallelization.
    """
    # :param start_time: the start time for the evolution, important when passing off a branch to a new worker
    #
    # :param psi_0: initial wavefunction for the evolution
    #
    # :param t_p: time parameters class
    #
    # :param fhm: Fermi-Hubbard operator class
    #
    # :param j_target: target current function for tracking control
    #
    # :param param: unscaled parameters class, needed when saving our expectations to file
    #
    # :param l: scaled parameters for constants used in the model
    #
    # :param obs: observables class, used to save our observables in memory
    #
    # :param branch_num: branch number for the arcsine in our control tracking scheme
    #
    # :param branch_string: string used to classify which branch we should be on
    #
    # :param split_check: a check to prevent newly spawned workers from immediately spawning another worker leading to
    #   infinitely many workers being spawned
    #
    # :results: creates files labelled with each parameter and which branch points it crossed by using a binary number
    #   with the following format: "0" for a kinking of the control field, "1" for a clean crossing of the control
    #   field.


    # assign initial wavefunction
    psi_t = psi_0

    # start evolution of wavefunction
    newtime = start_time
    tasks = []
    # print(start_time)
    while newtime < t_p.times[-1]:
        solver_args = dict(atol=1e-8)
        psi_t = evolve(v0=psi_t, t0=newtime, times=np.array([newtime + t_p.delta]),
                       f=original_tracking_evolution_with_branches, f_params=[fhm, j_target, l, branch_num],
                       **solver_args)

        # reshape the wavefunction to take expectations of it
        psi_t = psi_t.reshape(-1)

        # find control field at the given time and wavefunction
        phi = phi_J_track_with_branches(l, newtime + t_p.delta, j_target, fhm, psi_t, branch_num)

        # save expectations given wavefunction and control field
        obs.append_observables(psi_t, phi)

        # checks to see if the control field is sufficiently far away from the branch point to update the check
        theta = np.angle(obs.neighbour[-1])
        phi_p = ((-1)**(-branch_num))*(phi - theta - branch_num * PI)
        if split_check:
            if (PI / 2 - 5e-2) > phi_p > -(PI / 2 - 5e-2):
                split_check = False

        # check to see if the control field is near a branch point by using the formula
        #   phi - theta = Arcsin(X) = (-1)^(branch_number)*arcsin(X) + branch_number * pi
        #   to see if phi - theta is near half pi on its particular branch.
        else:
            if (PI/2 + 1e-2) > phi_p > (PI/2 - 1e-2):
                print("Branch point detected at t={}".format(newtime * l.freq))
                # make a copy of observables class and update the branch number and string.
                obs_jump = copy.deepcopy(obs)
                branch_num_jump = branch_num + 1
                branch_string_jump = branch_string + "+"
                branch_string += "0"

                # spawn new worker to run this function
                args = (newtime, psi_t, t_p, fhm, j_target, param, l, obs_jump, branch_num_jump, branch_string_jump,
                        True)
                # print("Spawned a new worker")
                split_check = True
                tasks.append(futures.submit(schrod_evol, *args))

            elif -(PI/2 + 1e-2) < phi_p < -(PI/2 - 1e-2):
                print("Branch point detected at t={}".format(newtime * l.freq))
                # make a deep copy of observables class and update the branch number and string.
                obs_jump = copy.deepcopy(obs)
                branch_num_jump = branch_num - 1
                branch_string_jump = branch_string + "-"
                branch_string += "0"

                # spawn new worker to run this function
                args = (newtime, psi_t, t_p, fhm, j_target, param, l, obs_jump, branch_num_jump, branch_string_jump,
                        True)
                # print("Spawned a new worker")
                split_check = True
                tasks.append(futures.submit(schrod_evol, *args))

        # update the evolution time
        # print(newtime)
        newtime += t_p.delta

    # when the loop is finished, save expectations to a file.
    expect = {}
    obs.save_observables(expect)

    savefile = 'scoop_branch_tracking/Data/{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbcF0={:.2f}-J_scale={:.2f}' \
               '/expectations:bstring={}.npz'.format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles,
                                                     t_p.n_steps, param.pbc, param.F0, param.J_scale, branch_string)
    print("Saving an expectation")
    expect['bstring'] = branch_string
    np.savez(savefile, **expect)
    assert all(task.result() for task in tasks), "Tasks did not exit properly"
    # for task in tasks:
    #     task.result()
    return True


if __name__ == "__main__":
    ti = time()
    print("Setup starting")
    # Generate our class for the unscaled parameters. These are primarily used for saving our data
    a_scale = 1.0
    param = unscaledparam(L=6, t0=0.52, U=0.5, pbc=True, field=32.9, F0=10, a=a_scale*4, a_scale=a_scale, J_scale=1,
                          tracking=1)

    # Generating our class of scaled parameters. This is used for most of the calculations
    lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0,
              F0=param.F0
              , a=param.a, pbc=param.pbc)

    # setup our evolution time parameters
    t_p = time_evolution_params(perimeter_params=lat, cycles=2, nsteps=int(2e3))

    # importing our data from preliminary simulation
    loadfile = 'Preliminary simulation/Data/expectations:{L}sites-{N_up}up-{N_down}down-{t0}t0-{U}U-{cycles}cycles-{n_steps}steps-{pbc}pbc.npz' \
        .format(**{**vars(t_p), **vars(param)})
    expectations = dict(np.load(loadfile))

    # interpolating our target current function with a cubic spline
    J_target = UnivariateSpline(t_p.times, param.J_scale * expectations['current'], s=0)

    # plt.figure('current')
    # plt.plot(t_p.times, J_target(t_p.times))
    # plt.plot(t_p.times, expectations['current'], ".")

    # setup quspin operators and lists
    FHM = Fermi_Hubbard(lat, t_p.cycles)

    print("Setup finished. It took {:.6f} seconds to initialize".format(time() - ti))

    # get inital energy and state
    ti = time()
    E, psi_0 = FHM.operator_dict['ham_init'].eigsh(k=1, which='SA')
    # print(psi_0)
    psi_t = psi_0.reshape(-1)
    print("Initial state and energy calculated. It took {:.2f} seconds to calculate".format(time() - ti))
    print("Ground state energy was calculated to be {:.2f}".format(E[0]))

    # initialize the observables class
    obser = observables(psi_t, J_target(0.0), phi_J_track(lat, 0.0, J_target, FHM, psi_t), FHM)

    # make a directory for this specific set of parameters
    path = "scoop_branch_tracking/Data/{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps" \
           "-{}pbcF0={:.2f}-J_scale={:.2f}".format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles,
                                                   t_p.n_steps, param.pbc, param.F0, param.J_scale)

    try:
        os.mkdir(path)
    except OSError as error:
        shutil.rmtree(path)
        os.mkdir(path)

    # run schrod_evol once to start the recursion
    schrod_evol(t_p.times[0], psi_t, t_p, FHM, J_target, param, lat, obser, branch_num=0.0, branch_string="",
                split_check=False)





