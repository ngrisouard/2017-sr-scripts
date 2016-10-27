# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:58:05 2016

@author: nicolas
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
# from numpy import shape
import netCDF4 as nc
import time
import getpass
import pickle
from matplotlib import rc
from pylab import get_current_fig_manager
if (sys.version_info > (3, 0)):  # Then Python 3 is running
    import ngobfftPy3 as tf
else:  # Then is has to be Python 2
    sys.path.append('/Users/NicoG/Dropbox/python_useful/obngfft')
    import ngobfft as tf

# %% NG start


class dictionary_to_class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def integrate_vol(x, z, u):
    """ avoiding loops divided the execution time by 13 """
    uint1 = np.trapz(u, z, axis=1)
    uint2 = np.trapz(uint1, x, axis=1)
    return uint2


def add_deriv_load(fID, uVar, deriv_to_load):
    """ computing the power of a given term
    fID is the file ID
    uVar is the component (uVar, vVar, wVar, bVar)
    deriv_to_load is the name (string) of the derivative at stake.
    Note that for bVar, conversion of derivative has to be done externally."""
    drvtv = fID.variables[deriv_to_load][:].copy()
    return uVar*drvtv


def add_deriv_four(x, uVar, K, n):
    """ Power of a term when x-derivatives are involved;
    Need to multiply by nu outside
    x: (1D) array along which the fft is performed
    uVar: the quantity to multiply and derive
    K: the (2D) array of wave numbers
    n: the order of derivative (hdegs)
    """
    FFF = tf.obfft(x, uVar, 2)
    k = tf.k_of_x(x)
    dnudxn = np.real(tf.obifft(k, FFF*(1j*K)**n, 2))
    return uVar*dnudxn


def print_elpsd(str_step, tic):
    print('{0} completed, elapsed time = {1: 4d} s'.format(str_step,
          int(time.time() - tic)))
    return time.time()


def print_int_elpsd(str_step, counter, ncounters, tic):
    print('    {0}: {1: >3d}% done, {2: 4d} s elapsed'.format(str_step,
          int((100*counter)/ncounters), int(time.time() - tic)))
    counter += 1
    return counter


def KEBTerms(s, setup, plst):

    Var = {}
    print('Start loading.')

    tic = time.time()

    if s.np == 1:
        fID_XZ = nc.Dataset(setup+'/2D/XZ.nc', 'r')
        fID_d = nc.Dataset(setup+'/2D/derivs.nc', 'r')
        fID_td = nc.Dataset(setup+'/2D/t_derivs.nc', 'r')

        t = fID_XZ.variables['tVar'][:].copy()
        for vv in ['u', 'v', 'w', 's1']:
            Var[vv] = fID_XZ.variables[vv+'Var'][:, :, :].copy()
            tic = print_elpsd('loading of '+vv, tic)
    else:
        fID_XZ = {}
        fID_d = {}
        fID_td = {}
        for ii in range(s.np):
            fID_XZ[str(ii)] = nc.Dataset('{0}/2D/XZ_{1:03d}.nc'.format(
                                         setup, ii), 'r')
            fID_d[str(ii)] = nc.Dataset('{0}/2D/derivs_{1:03d}.nc'.format(
                                             setup, ii), 'r')
            fID_td[str(ii)] = nc.Dataset('{0}/2D/t_derivs_{1:03d}.nc'.format(
                                         setup, ii), 'r')

        t = fID_XZ['0'].variables['tVar'][:].copy()
        for vv in ['u', 'v', 'w', 's1']:
            utmp = fID_XZ['0'].variables[vv+'Var'][:, :, :].copy()
            for jj in range(1, s.np):
                to_cat = fID_XZ[str(jj)].variables[vv+'Var'][:, :, :].copy()
                utmp = np.concatenate((utmp, to_cat), axis=1)
            Var[vv] = utmp

    # tic = print_elpsd('Loading Vars', tic)

    # %%
    # Conversion to PE calculation

    print('Shape of RLoc = {}'.format(s.RLoc.shape))

    conv = s.g*(Var['s1'][:, :, :] - s.RLoc[:, :])*Var['w']/s.rho_0

    # conv = - Var['b']*Var['w']

    Ctot = integrate_vol(s.x[plst[1]:plst[0]+1], s.z[plst[2]:],
                         conv[:, plst[2]:, plst[1]:plst[0]+1])

    tic = print_elpsd('Conversion term', tic)

    # %%
    # Flux calculations

    if s.np > 1:
        raise NameError("We don't have a procedure for the flux with np>1")

    pVar_xplus = fID_d.variables['pVar'][:, :, plst[0]].copy()
    phi_xplus = pVar_xplus*Var['u'][:, :, plst[0]]  # [kg/s^3]or[W/m^2]

    pVar_xminus = fID_d.variables['pVar'][:, :, plst[1]].copy()
    phi_xminus = -pVar_xminus*Var['u'][:, :, plst[1]]

    pVar_zminus = fID_d.variables['pVar'][:, plst[2], :].copy()
    phi_zminus = -pVar_zminus*Var['w'][:, plst[2], :]

    # Zero flux occurs through top surface

    # integrate and add the three paths of energy flux (taking CW path)
    PHI1 = np.trapz(phi_xplus[:, plst[2]:], s.z[plst[2]:], axis=1)
    PHI2 = np.trapz(phi_xminus[:, plst[2]:], s.z[plst[2]:], axis=1)
    PHI3 = np.trapz(phi_zminus[:, plst[1]:plst[0]+1], s.x[plst[1]:plst[0]+1],
                    axis=1)

    tic = print_elpsd('Fluxes', tic)

    # %%
    # Geostrophic Shear Production (GSP)

    GSP = 0.*Var['u']
    GSP[:, :, :] = s.S2Loc[:, :]*Var['v'][:, :, :]*Var['w'][:, :, :]/s.f  # +
    #                         Var['u'][:, :, :]*b[:, :, :]/N2[:, :])
    # NG just removed the contribution from PE

    GSPtot = integrate_vol(s.x[plst[1]:plst[0]+1], s.z[plst[2]:],
                           GSP[:, plst[2]:, plst[1]:plst[0]+1])

    tic = print_elpsd('GSP', tic)

    # %%
    # Lateral Shear Production (LSP)

    LSP = 0.*Var['u']
    LSP[:, :, :] = s.f*s.RoGLoc[:, :]*Var['v'][:, :, :]*Var['u'][:, :, :]

    LSPtot = integrate_vol(s.x[plst[1]:plst[0]+1], s.z[plst[2]:],
                           LSP[:, plst[2]:, plst[1]:plst[0]+1])

    tic = print_elpsd('LSP', tic)

    # %% NG removed PE
    # Change in KE and NOT PE with time

    # Calculating dKE/dt at every grid point within CV grid
    dtKE = 0.*Var['u']
    counter = 1

    for ii in ['u', 'v', 'w']:
        dtKE += add_deriv_load(fID_td, Var[ii], 'd{}dt'.format(ii))
        counter = print_int_elpsd('dKEdt', counter, 3, tic)

    dtKEtot = integrate_vol(s.x[plst[1]:plst[0]+1], s.z[plst[2]:],
                            dtKE[:, plst[2]:, plst[1]:plst[0]+1])

    dEdt = dtKEtot  # +dtPEtot

    tic = print_elpsd('dEdt', tic)
    # %%
    # Compute dissipation terms

    k = tf.k_of_x(s.x)
    K, M = np.meshgrid(k, s.z)
    # dffs = {'nuh2': s.nuh, 'nuh4': -s.nuh2, 'nuz2': s.nuz, 'nuz4': -s.nuz2}

    ldiss = 0.*Var['u']  # Laplacian dissipation
    hdiss = 0.*Var['u']  # hyperviscous dissipation
    counter = 1
    print('Start Dissipation')

    for ii in ['u', 'v', 'w']:
        # dzNm = 'd2'+ii+'dz2'
        ldiss += s.nuz * add_deriv_load(fID_d, Var[ii], 'd2'+ii+'dz2')
        counter = print_int_elpsd('Dissipation', counter, 12, tic)
        ldiss += s.nuh * add_deriv_four(s.x, Var[ii], K, 2)
        counter = print_int_elpsd('Dissipation', counter, 12, tic)
        hdiss += -s.nuz2 * add_deriv_load(fID_d, Var[ii], 'd4'+ii+'dz4')
        counter = print_int_elpsd('Dissipation', counter, 12, tic)
        hdiss += -s.nuh2 * add_deriv_four(s.x, Var[ii], K, 4)
        counter = print_int_elpsd('Dissipation', counter, 12, tic)

    # dissipation of the thermal wind
    # d2ThWdz2 = s.S2Loc / (s.f*s.dltH*s.Lz)
    # dissip = dissip[:, :, :]  # + (nuz*d2ThWdz2[:, :])*Var['v'][:, :, :]

    LDtot = integrate_vol(s.x[plst[1]:plst[0]+1], s.z[plst[2]:],
                          ldiss[:, plst[2]:, plst[1]:plst[0]+1])
    HDtot = integrate_vol(s.x[plst[1]:plst[0]+1], s.z[plst[2]:],
                          hdiss[:, plst[2]:, plst[1]:plst[0]+1])

    tic = print_elpsd('Dissipation', tic)

    # %%
    # Final step: place everything in a dictionary and save it
    os.system('rm {0}/KEBterms.npz'.format(setup))
    np.savez('{0}/KEBterms.npz'.format(setup),
             t=t,
             pUpls=PHI1,
             pUmns=PHI2,
             pWmns=PHI3,
             GSP=GSPtot,
             LSP=LSPtot,
             dKEdt=dEdt,
             LDiss=LDtot,
             HDiss=HDtot,
             toPE=Ctot)

    dic_of_terms = {
        't': t,
        'pUpls': PHI1,
        'pUmns': PHI2,
        'pWmns': PHI3,
        'GSP': GSPtot,
        'LSP': LSPtot,
        'dKEdt': dEdt,
        'LDiss': LDtot,
        'HDiss': HDtot,
        'toPE': Ctot
        }

    return dic_of_terms

    # %%

    #
    #
    # Full accounting of Energy Budget -> checking for LHS=RHS and all energy
    # is conserved
    #
    #

    #    Ebudget = -PHI - GSPtot - LSPtot - dEdt + LDtot + HDtot - Ctot
    #
    #    plt.figure()
    #    plt.plot(t, -PHI1, 'k', label='x+ Flux')
    #    plt.plot(t, -PHI2, 'k--', label='x- Flux')
    #    plt.plot(t, -PHI3, 'k+-', label='z- Flux')
    #    plt.plot(t, -GSPtot, 'm', label='GSP')
    #    plt.plot(t, -LSPtot, 'm--', label='LSP')
    #    plt.plot(t, -dEdt, 'b', label='dEdt')
    #    plt.plot(t, Dtot, 'g', label='Dissipation')
    #    plt.plot(t, -Ctot, 'c', label='-Conversion')
    #    plt.plot(t, Ebudget, 'r--', label='Cumulative Budget', lw=2)
    #
    #    plt.xlabel("Time [$s$]")
    #    plt.ylabel("Integrated Energy Per Unit Density [$m^4/s^3$]")
    #    plt.title("Full Accounting of Energy Budget")
    #    plt.grid()
    #    plt.legend(loc='upper left')
    #    plt.show()

    if s.np == 1:
        fID_XZ.close()
        fID_d.close()
        fID_td.close()
    else:
        for ii in range(s.np):
            fID_XZ[str(ii)].close()
            fID_d[str(ii)].close()
            fID_td[str(ii)].close()

    time_tot = time.time() - tic_strt
    mins = int(time_tot/60)
    secs = int(time_tot - 60*mins)

    print(' ')
    print('          ******')
    print(' ')
    print('All done! Elapsed time = {0:3d}:{0:02d}'.format(mins, secs))
    print(' ')
    print('          ******')
    print(' ')


def plot_one_budget(KET, s, nm, t_strt):

    rc('text', usetex=True)
    font = {'size': 26}
    rc('font', **font)
    rc('lines', linewidth=2)
    # lwd=2

    RES = (KET['dKEdt'] + KET['pUpls'] + KET['pUmns'] + KET['pWmns'] +
           KET['toPE'] + KET['GSP'] + KET['LSP'] - KET['LDiss'] - KET['HDiss'])

    # fig = plt.figure(nm)
    # plt = fig.gca()
    t_scl = 0.5*KET['t'][t_strt:]*s.f/np.pi
    plt.plot(t_scl, KET['dKEdt'][t_strt:], 'k',
             label='$\partial KE/\partial t$')
    plt.plot(t_scl, KET['GSP'][t_strt:], 'b', label='Geostrophic SP')
    plt.plot(t_scl, KET['LSP'][t_strt:], 'b--', label='Lateral SP')
    plt.plot(t_scl, KET['toPE'][t_strt:], color='brown', label="$-wb'$")
    plt.plot(t_scl, KET['pUpls'][t_strt:], 'r:', label="$p'u|_{right}$")
    plt.plot(t_scl, KET['pUmns'][t_strt:], 'r', label="$p'u|_{le\!f\!t}$")
    plt.plot(t_scl, KET['pWmns'][t_strt:], 'r--', label="$p'w|_{bottom}$")
    plt.plot(t_scl, KET['LDiss'][t_strt:], 'g',
             label=r"$-(\nu \nabla^2 \vec v')\cdot \vec v'$")
    plt.plot(t_scl, KET['HDiss'][t_strt:], 'g--',
             label=r"$-(-\nu_4 \nabla^4 \vec v')\cdot \vec v'$")
    # plt.plot(t_scl, RES[t_strt:], 'r--', label="Residual")

    plt.grid('on')
    plt.axis([t_scl[0], 12., -6e-4, 6e-4])
    plt.xlabel("$tf/2\pi$")
    plt.ylabel("Power Per Unit Density [$m^4/s^3$]")
    # plt.title("")
    plt.legend(loc='upper right')
    # plt.tight_layout()
    plt.show()

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    get_current_fig_manager().window.raise_()
    # mng.frame.Maximize(True)


# %%

tic_strt = time.time()

if getpass.getuser() == 'NicoG':
    base = '/Volumes/LaCie2TB/mbf/'
    # base = '/Users/NicoG/FS_Morgan/outputs/'
elif getpass.getuser() == 'nicolas':
    base = '/data/nicolas/mbf/'
elif getpass.getuser() == 'mbfox':
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the base directory?')

plt.close('all')
boundaries_list = [850, 713, 280]  # right, left and bottom boundaries

save = 0  # 1 to compute and save, 0 to just load

plt.close('all')

for ii in ['01']:
    for iN in ['68']:
        for iF in ['12']:  # , '12', '14']:
            for iR in ['12']:
                setup = 'N' + iN + 'F' + iF + 'R' + iR + '-' + ii
                root_dir = base + setup

                print(root_dir)

                pkl_file = open(root_dir+'/spec_params.pkl', 'rb')
                d2pkl = pickle.load(pkl_file)
                pkl_file.close()

                s = dictionary_to_class(**d2pkl)
                del d2pkl

                if save:
                    KET = KEBTerms(s, root_dir, boundaries_list)
                else:
                    KET = np.load(root_dir+'/KEBterms.npz')

                dic_of_starts = {
                    setup[:-3]: 50
                    }

                plot_one_budget(KET, s, setup[:-3], dic_of_starts[setup[:-3]])
