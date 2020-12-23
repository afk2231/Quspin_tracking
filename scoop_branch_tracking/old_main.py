from __future__ import print_function, division
import os
import sys
from quspin.operators import hamiltonian
from quspin.tools.evolution import evolve
from classes.fermi_hubbard_class.setup_fermi_hubbard import Fermi_Hubbard
from classes.perimeter_params.tools import parameter_instantiate as hhg  # Used for scaling units.
from classes.unscaled_parameters.unscaledparam import unscaledparam
from classes.time_param.t_param import time_evolution_params
from classes.observables_class.observables import observables
from functions.original_tracking_equations.original_tracking_equations import phi_J_track_with_branches
from functions.original_tracking_equations.original_tracking_equations import original_tracking_evolution_with_branches
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import psutil
import numpy as np  # general math functions
from time import time  # tool for calculating computation time
from matplotlib import pyplot as plt
from scoop import futures
import h5py as h5



def scoopfunction(newtime, t_p, psi, fhm, j_target, l, branch_num, branch_string, psigroup):
    """
    Uses Schrodinger equation to find the wavefunction at a time step forward and appends it to a greater
    wavefunction array. Function stops if it has evolved to the target endpoint or if the wavefunction reproduces
    a control field value near the branch points at phi - theta = (2k+1)*pi/2. Returns the current time, wavefunction,
    and how to switch the branch number in the jumped worker.

    Base level function to use with scoop in order to find the wavefunction for all possible branch decisions
    for a given target current function.

    This implementation uses quspin's architecture for the wavefunction of a Fermi-Hubbard lattice.
    """

#   :param newtime: the current time step of the evolution

#   :param t_p: the class which holds our lattice of time, number of steps, and time lattice size.

#   :param psi: the wavefunction for the Fermi Hubbard lattice

#   :param fhm: class for operators of our Fermi Hubbard lattice using quspin's architecture

#   :param j_target: UnivariateSpline of our target current function.

#   :param l: perimeter parameters used in our fermi hubbard model

#   :param branch_num: the branch number of the arcsine within the control field

    # start the while loop corresponding to getting the wavefunction at each of the points defined in times
    while newtime < t_p.times[-1]:
        # solve Schrodinger equation using the original arcsine tracking equation for the wavefunction at time =
        #   t + delta
        solver_args = dict(atol=1e-12)
        print(psi)
        psi_t = evolve(v0=psi[-1], t0=newtime, times=np.array([newtime + t_p.delta]),
                       f=original_tracking_evolution_with_branches, f_params=[fhm, j_target, l, branch_num],
                       **solver_args)

        # append the new wavefunction at t + delta
        psi.append(psi_t)

        # reshape the wavefunction in order to take expectations using it
        psi_t = psi_t.reshape(-1)

        # update the time for the next step
        newtime = newtime + t_p.delta

        # get the control field given the wavefunction and the target current
        phi = phi_J_track_with_branches(l, newtime + t_p.delta, j_target, fhm, psi_t, branch_num)

        # get nearest neighbour expectation for the branch check
        neighbour = fhm.operator_dict['hop_left_op'].expt_value(psi_t)

        # checks if phi - theta = Arcsin(X) = (-1)^(branch_number)*arcsin(X) + branch_number * pi is near an odd
        #   multiple of pi/2. if so, return the function with the current time, all wavefunctions from previous times,
        #   and which direction should the branch switch.
        if ((-1)**branch_num)*(phi - np.angle(neighbour) - branch_num * np.pi) > (np.pi/2 - 1e-2):
            return [newtime, psi, +1]
        elif ((-1)**branch_num)*(phi - np.angle(neighbour) - branch_num * np.pi) < -(np.pi/2 - 1e-2):
            return [newtime, psi, -1]

    # if the current worker doesn't find a branch point return the function with the current time, all wavefunctions
    # previously found, and no direction of branch switching.
    return [newtime, psi, 0]

def scoopfunction2(newtime, t_p, psi, fhm, j_target, l, branch_num, branch_string, psigroup):
    """
    Recursive function to find all control fields given a target current function. Ends when scoopfunction returns no
    branch switching needed and assigns a dataset to the given branch decisions it has made in it's past; otherwise,
    it spawns two new workers to continue until scoopfunction returns no branch switching needed.

    Second level function used with scoop, calls the base level function: scoopfunction, and itself.

    This implementation uses my segmented classes for operator and parameter storage; however, in the future I'd like
    to adapt this to a universal class that actually does stuff like take expectations of specific operators.
    """
#   :param newtime: the current time step of the evolution

#   :param t_p: the class which holds our lattice of time, number of steps, and time lattice size.

#   :param psi: the wavefunction for the Fermi Hubbard lattice

#   :param fhm: class for operators of our Fermi Hubbard lattice using quspin's architecture

#   :param j_target: UnivariateSpline of our target current function.

#   :param l: perimeter parameters used in our fermi hubbard model

#   :param branch_num: the branch number of the arcsine within the control field

#   :param branch_string: string of binary corresponding to the number of branches jumped (1) or bounced off (0)

#   :param psigroup: h5py group for psi used to catalog all the control fields

    # spawn task according to the passed keyword arguments
    args = (newtime, t_p, psi, fhm, j_target, l, branch_num, branch_string, psigroup)
    task = futures.submit(scoopfunction, *args)

    # get results of the spawned worker
    time, psi, branchjump = task.result()

    # checks to see if the worker finished the code
    if branchjump == -1 or branchjump == +1:
        # spawns two new workers with the same times and wave functions.

        # make a shallow copy of the kwargs for kwargs jump since we don't want a deep copy of the FHM class
        args_jump = args.copy()
        args_jump['branch_num'] += branchjump*np.pi
        args_jump['newtime'] = time
        args['newtime'] = time
        args['psi'] = psi.copy()
        args_jump['psi'] = psi.copy()
        args['branch_string'] += "0"
        args_jump['branch_string'] += "1"

        # spawn a worker that will jump the branchpoint
        task_jump = futures.submit(scoopfunction2, *args_jump)

        # spawn another worker that will stay on the current branch
        task_nojump = futures.submit(scoopfunction2, *args)
    else:
        # create h5py subgroup corresponding to the branch decisions chosen
        branchsubgroup = args['psigroup'].create_group(args['branch_string'])

        # save the final wavefunction for the corresponding branch decisions to a h5py dataset.
        dset = branchsubgroup.create_dataset("psis", data=psi)


if __name__ == "__main__":
    ti = time()
    print("Setup starting")
    # Generate our class for the unscaled parameters. These are primarily used for saving our data
    param = unscaledparam(L=6, t0=0.52, U=0, pbc=True, field=32.9, F0=3, a=4, a_scale=1, J_scale=1, tracking=1)

    # Generating our class of scaled parameters. This is used for most of the calculations
    lat = hhg(field=param.field, nup=param.N_up, ndown=param.N_down, nx=param.L, ny=0, U=param.U, t=param.t0,
              F0=param.F0
              , a=param.a, pbc=param.pbc)

    # setup our evolution time parameters
    t_p = time_evolution_params(perimeter_params=lat, cycles=1, nsteps=int(2e3))

    # importing our data from preliminary simulation
    loadfile = 'Preliminary simulation/Data/expectations:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc.npz' \
        .format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc)
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

    # creating our filename string for h5py
    myFile_name = "data:{}sites-{}up-{}down-{}t0-{}U-{}cycles-{}steps-{}pbc-{}F0.h5"\
        .format(param.L, param.N_up, param.N_down, param.t0, param.U, t_p.cycles, t_p.n_steps, param.pbc, param.F0)

    # opening h5py file to append data.
    with h5.File(myFile_name, "w") as f:
        # create psi subgroup for data collection
        psisubgroup = f.create_group("psi")

        # start recursion for all branch points
        scoopfunction2(newtime=0.0,
                       t_p=t_p,
                       psi=[psi_t,],
                       fhm=FHM,
                       j_target=J_target,
                       l=lat,
                       branch_num=0.0,
                       branch_string='',
                       psigroup=psisubgroup)
