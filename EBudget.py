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
    sys.path.append('/Users/NicoG/Dropbox/python_useful/obngfft')
    import ngobfftPy3 as tf
else:  # Then is has to be Python 2
    sys.path.append('/Users/NicoG/Dropbox/python_useful/obngfft')
    import ngobfft as tf
# %% NG start -----------------------------------------------------------------


class dictionary_to_class:
    def __init__(self, **entries):
        self.__dict__.update(entries)
# %% --------------------------------------------------------------------------


def int_vol(dx, dz, u):
    uint1 = np.trapz(u, dx=dz, axis=1)
    uint2 = np.trapz(uint1, dx=dx, axis=1)
    return uint2
# %% --------------------------------------------------------------------------


def add_deriv_load(fID, uVar, to_load, kbott, plst):
    """ Computing the power of a given term whose derivative is stored.
    fID is the file ID
    uVar is the component (uVar, vVar, wVar, bVar)
    deriv_to_load is the name (string) of the derivative at stake.
    Note that for bVar, conversion of derivative has to be done externally.
    """
    drvtv = fID.variables[to_load][plst[3]:, kbott:, plst[1]:plst[0]+1].copy()
    return uVar*drvtv
# %% --------------------------------------------------------------------------


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
# %% --------------------------------------------------------------------------


def print_elpsd(str_step, tic):
    print('{0} completed, elapsed time = {1: 4d} s'.format(str_step,
          int(time.time() - tic)))
    return time.time()
# %% --------------------------------------------------------------------------


def print_int_elpsd(str_step, counter, ncounters, tic):
    print('    {0}: {1: >3d}% done, {2: 4d} s elapsed'.format(str_step,
          int((100*counter)/ncounters), int(time.time() - tic)))
    counter += 1
    return counter
# %% --------------------------------------------------------------------------


def MEBTerms(s, setup, plst):  # Mechanichal Energy Budget Terms

    Var = {}  # this dictionary will be the recipient of the loaded variables

    tic = time.time()
    tic_strt = time.time()

    # these dictionaries will be the recipients of the file IDs
    fID_XZ = {}
    fID_d = {}
    fID_td = {}

    # locate the processor containing the bottom of the box
    npb = plst[2]//(s.nz//s.np)

    ntint = s.nites//s.sk_it + 1 - plst[3]
    nxint = plst[0] - plst[1] + 1
    if s.np > 1:
        nzint = (s.np - npb) * (s.nz//s.np) + 1
    else:
        nzint = s.nnz - plst[2]

    print(nzint)

    ts_shp = (ntint, )  # shape of each time series
    PHIp = np.zeros(ts_shp)
    PHIm = np.zeros(ts_shp)
    PHIz = np.zeros(ts_shp)
    AdPEtot = np.zeros(ts_shp)
    GSPtot = np.zeros(ts_shp)
    LSPtot = np.zeros(ts_shp)
    LCtot = np.zeros(ts_shp)
    dtEtot = np.zeros(ts_shp)
    LDztot = np.zeros(ts_shp)
    HDztot = np.zeros(ts_shp)
    LDhtot = np.zeros(ts_shp)
    HDhtot = np.zeros(ts_shp)

    to_int_shp = (ntint, nzint, nxint)
    phi_xpls = np.zeros((ntint, nzint))
    phi_xmns = np.zeros((ntint, nzint))
    phi_zmns = np.zeros((ntint, nxint))
    AdPE = np.zeros(to_int_shp)
    GSP = np.zeros(to_int_shp)
    LSP = np.zeros(to_int_shp)
    LC = np.zeros(to_int_shp)
    dtE = np.zeros(to_int_shp)
    LDz = np.zeros(to_int_shp)
    HDz = np.zeros(to_int_shp)
    LDh = np.zeros((ntint, nzint, s.nx))
    HDh = np.zeros((ntint, nzint, s.nx))

    k = tf.k_of_x(s.x)

    # open and read for each file
    for pr in range(npb, s.np):

        print(' ')
        print('---------------- Proc #{0:02d}/{1:02d} ----------------'.format(
              pr-npb+1, s.np-npb))
        print(' ')

        prs = '{0:02d}'.format(pr)  # string version of pr, 2 digits

        if s.np > 1:
            kb = pr*s.nz//s.np  # bottom z index of one proc: proc# * pts/proc
            kt = (pr+1)*s.nz//s.np  # top z index of one proc
            kbi = (pr-npb)*s.nz//s.np  # bottom z index for integrands
            kti = (pr-npb+1)*s.nz//s.np  # top z index for integrands
            if pr == s.np-1:
                kt += 1
                kti += 1
        else:  # then these numbers have to be the CV bounds
            kb = plst[2]
            kt = s.nnz
            kbi = 0  # bottom z index of one proc for integrands
            kti = nzint  # top z index of one proc for integrands

        if s.np == 1:  # string version of pr, as suffix of file name
            npstr = ''
        else:
            npstr = '_0' + prs

        # bottom boundary
        if s.np > 1:  # bottom of a CV is the bottom of a processor domain
            kbott = 0
        else:  # bottom of CV is the actual bottom of CV as prescribed
            kbott = plst[2]

        # %% Loading variables ------------------------------------------------
        print('Start loading, proc #' + prs)

        fID_XZ[prs] = nc.Dataset(setup + '/2D/XZ' + npstr + '.nc', 'r')
        fID_d[prs] = nc.Dataset(setup + '/2D/derivs' + npstr + '.nc', 'r')
        fID_td[prs] = nc.Dataset(setup + '/2D/t_derivs' + npstr + '.nc', 'r')

        # print(setup + '/2D/XZ' + npstr + '.nc')
        # print(fID_d[prs].variables.keys())

        for vv in ['u', 'v', 'w', 's1']:
            Var[vv] = (fID_XZ[prs].variables[vv+'Var'][plst[3]:, kbott:,
                       :].copy())
            tic = print_elpsd('loading ' + vv + ', proc #' + prs, tic)

        bVar = -s.g / s.rho_0 * (Var['s1'][:, :, :] - s.RLoc[kb:kt, :])
        bN2 = bVar / s.N2Loc[kb:kt, :]

        if pr == npb:
            t = fID_XZ[prs].variables['tVar'][plst[3]:].copy()

        # tic = print_elpsd('Loading Vars', tic)

        # %% ------------------------------------------------------------------
        # Flux calculations

        if pr == npb:  # works no matter how many procs there are
            pVar_zmns = (fID_d[prs].variables['pVar'][plst[3]:, kbott,
                         plst[1]:plst[0]+1].copy())
            phi_zmns = -(pVar_zmns + 0.5 * bVar[:, 0, plst[1]:plst[0]+1] *
                         bN2[:, 0, plst[1]:plst[0]+1]
                         ) * Var['w'][:, 0, plst[1]:plst[0]+1]
            # the - sign above is because of orientation of outward normal

        # right boundary
        pVar_xpls = (fID_d[prs].variables['pVar'][plst[3]:, kbott:,
                     plst[0]].copy())
        phi_xpls[:, kbi:kti] = (pVar_xpls +
                                0.5 * bVar[:, :, plst[0]] * bN2[:, :, plst[0]]
                                )*Var['u'][:, :, plst[0]]
        # u, v, etc. were only loaded from plst[3]

        # left boundary
        pVar_xmns = (fID_d[prs].variables['pVar'][plst[3]:, kbott:,
                     plst[1]].copy())
        phi_xmns[:, kbi:kti] = -(pVar_xmns +
                                 0.5 * bVar[:, :, plst[1]] * bN2[:, :, plst[1]]
                                 )*Var['u'][:, :, plst[1]]
        # the - sign above is because of orientation of outward normal

        # Zero flux occurs through top surface

        tic = print_elpsd('Fluxes', tic)

        # %% ------------------------------------------------------------------
        # Advection of PE

        # cf. Notability sheet
        dN2dx = s.S2Loc[kb:kt, plst[1]:plst[0]+1] / (s.dltH * s.Lz)
        dN2dz = s.d2Rdz2[kb:kt, plst[1]:plst[0]+1] * (- s.g / s.rho_0)
        AdPE[:, kbi:kti, :] = (0.5 * bN2[:, :, plst[1]:plst[0]+1]**2 *
                               (Var['u'][:, :, plst[1]:plst[0]+1] * dN2dx +
                               Var['w'][:, :, plst[1]:plst[0]+1] * dN2dz))

        tic = print_elpsd('PE advection', tic)

        # %% ------------------------------------------------------------------
        # Geostrophic Shear Production (GSP)

        # GSP = 0.*Var['u']
        GSP[:, kbi:kti, :] = (Var['v'][:, :, plst[1]:plst[0]+1] *
                              Var['w'][:, :, plst[1]:plst[0]+1] *
                              s.S2Loc[kb:kt, plst[1]:plst[0]+1] / s.f)

        tic = print_elpsd('GSP', tic)

        # %% ------------------------------------------------------------------
        # Lateral Conversion (LC)

        LC[:, kbi:kti, :] = (Var['u'][:, :, plst[1]:plst[0]+1] *
                             bN2[:, :, plst[1]:plst[0]+1] *
                             s.S2Loc[kb:kt, plst[1]:plst[0]+1])

        tic = print_elpsd('LC', tic)

        # %% ------------------------------------------------------------------
        # Lateral Shear Production (LSP)

        LSP[:, kbi:kti, :] = (Var['v'][:, :, plst[1]:plst[0]+1] *
                              Var['u'][:, :, plst[1]:plst[0]+1] *
                              s.f * s.RoGLoc[kb:kt, plst[1]:plst[0]+1])

        tic = print_elpsd('LSP', tic)

        # %% ------------------------------------------------------------------
        # Compute dissipation terms and dEdt

        K, M = np.meshgrid(k, s.z[kb:kt])

        counter = 1
        print('Proc #' + prs + ', start dissipation')

        for ii in ['u', 'v', 'w']:
            # dzNm = 'd2'+ii+'dz2'

            dotProd = Var[ii][:, :, plst[1]:plst[0]+1]
            to_four_deriv = Var[ii]

            dtE[:, kbi:kti, :] += add_deriv_load(fID_td[prs], dotProd,
                                                 'd'+ii+'dt', kbott, plst)

            LDz[:, kbi:kti, :] += add_deriv_load(fID_d[prs], dotProd,
                                                 'd2'+ii+'dz2', kbott, plst)
            LDh[:, kbi:kti, :] += add_deriv_four(s.x, to_four_deriv, K, 2)
            counter = print_int_elpsd('Dissipation', counter, 8, tic)

            HDz[:, kbi:kti, :] -= add_deriv_load(fID_d[prs], dotProd,
                                                 'd4'+ii+'dz4', kbott, plst)
            HDh[:, kbi:kti, :] -= add_deriv_four(s.x, to_four_deriv, K, 4)
            counter = print_int_elpsd('Dissipation', counter, 8, tic)

        dtE[:, kbi:kti, :] += (add_deriv_load(fID_td[prs],
                               -s.g * bN2[:, :, plst[1]:plst[0]+1] / s.rho_0,
                               'ds1dt', kbott, plst))

        LDz[:, kbi:kti, :] += (add_deriv_load(fID_d[prs],
                               -s.g * bN2[:, :, plst[1]:plst[0]+1] / s.rho_0,
                               'd2s1dz2', kbott, plst))
        LDh[:, kbi:kti, :] += add_deriv_four(s.x, bVar, K, 2)/s.N2Loc[kb:kt, :]
        counter = print_int_elpsd('Dissipation', counter, 8, tic)

        HDz[:, kbi:kti, :] -= (add_deriv_load(fID_d[prs],
                               -s.g*bN2[:, :, plst[1]:plst[0]+1] / s.rho_0,
                               'd4s1dz4', kbott, plst))
        HDh[:, kbi:kti, :] -= add_deriv_four(s.x, bVar, K, 4)/s.N2Loc[kb:kt, :]
        counter = print_int_elpsd('Dissipation', counter, 8, tic)

        # dissipation of the thermal wind
        # d2TWdz2 = s.S2Loc / (s.f*s.dltH*s.Lz)
        # LDh += - s.g / s.rho_0 * s.d2Rdx2[kb:kt, :] * bN2[:, :, :]
        #               # + s.d2TWdx2[kb:kt, :] * Var['v'][:, :, :])
        # LDz += - s.g / s.rho_0 * s.d2Rdz2[kb:kt, :] * bN2[:, :, :]
        # + d2TWdz2[kb:kt, :] * Var['v'][:, :, :]

        tic = print_elpsd('Dissipation and dEdt', tic)

        # %% ------------------------------------------------------------------
        fID_XZ[prs].close()
        fID_d[prs].close()
        fID_td[prs].close()

    # %%
    # %% Integrations
    # integrate and add the three paths of energy flux (taking CW path)
    PHIm = np.trapz(phi_xmns, dx=s.dz, axis=1)
    PHIp = np.trapz(phi_xpls, dx=s.dz, axis=1)
    PHIz = np.trapz(phi_zmns, dx=s.dx, axis=1)

    # volume integrations
    AdPEtot = int_vol(s.dx, s.dz, AdPE)
    GSPtot = int_vol(s.dx, s.dz, GSP)
    LCtot = int_vol(s.dx, s.dz, LC)
    LSPtot = int_vol(s.dx, s.dz, LSP)
    dtEtot = int_vol(s.dx, s.dz, dtE)
    LDztot = int_vol(s.dx, s.dz, s.nuz * LDz)
    HDztot = int_vol(s.dx, s.dz, s.nuz2 * HDz)
    LDhtot = int_vol(s.dx, s.dz, s.nuh * LDh[:, :, plst[1]:plst[0]+1])
    HDhtot = int_vol(s.dx, s.dz, s.nuh2 * HDh[:, :, plst[1]:plst[0]+1])

    tic = print_elpsd('Integrations', tic)

    # %%
    # Final step: place everything in a dictionary and save it
    os.system('rm {0}/MEBterms.npz'.format(setup))
    np.savez('{0}/MEBterms.npz'.format(setup),
             t=t,
             pUpls=PHIp,
             pUmns=PHIm,
             pWmns=PHIz,
             AdPE=AdPEtot,
             GSP=GSPtot,
             LSP=LSPtot,
             LC=LCtot,
             dEdt=dtEtot,
             LDz=LDztot,
             HDz=HDztot,
             LDh=LDhtot,
             HDh=HDhtot)

    dic_of_terms = {'t': t,
                    'pUpls': PHIp,
                    'pUmns': PHIm,
                    'pWmns': PHIz,
                    'AdPE': AdPEtot,
                    'GSP': GSPtot,
                    'LSP': LSPtot,
                    'LC': LCtot,
                    'dEdt': dtEtot,
                    'LDz': LDztot,
                    'HDz': HDztot,
                    'LDh': LDhtot,
                    'HDh': HDhtot}

    # %%

    time_tot = time.time() - tic_strt
    mins = int(time_tot/60)
    secs = int(time_tot - 60*mins)

    print(' ')
    print('          ***********')
    print(' ')
    print('All done! Elapsed time = {0:3d}:{0:02d}'.format(mins, secs))
    print(' ')
    print('          ***********')
    print(' ')

    # %%
    return dic_of_terms


# %%
def plot_one_budget(MET, s, nm, t_strt):

    # rc('text', usetex=True)
    font = {'size': 16}
    rc('font', **font)
    rc('lines', linewidth=2)
    # lwd=2

    RES = (MET['dEdt'] + MET['pUpls'] + MET['pUmns'] + MET['pWmns'] +
           MET['LC'] + MET['AdPE'] + MET['GSP'] + MET['LSP'] -
           MET['LDz'] - MET['HDz'] - MET['LDh'] -
           MET['HDh'])

    # fig = plt.figure(nm)
    # plt = fig.gca()
    plt.figure(1)
    t_scl = 0.5*MET['t'][t_strt:]*s.f/np.pi
    plt.plot(t_scl, MET['dEdt'][t_strt:], 'k',
             label='$\partial E/\partial t$')
    plt.plot(t_scl, MET['GSP'][t_strt:], 'b', label='Geostrophic SP')
    plt.plot(t_scl, MET['LSP'][t_strt:], 'b--', label='Lateral SP')
    plt.plot(t_scl, MET['LC'][t_strt:], color='brown', label="$M^2ub'/N^2$")
    plt.plot(t_scl, MET['AdPE'][t_strt:], color='orange', label="PE advection")
    plt.plot(t_scl, MET['pUpls'][t_strt:], 'm:',
             label="$[p' + b'^2/(2N^2)]u|_{right}$")
    plt.plot(t_scl, MET['pUmns'][t_strt:], 'm',
             label="$-[p' + b'^2/(2N^2)]u|_{le\!f\!t}$")
    plt.plot(t_scl, MET['pWmns'][t_strt:], 'm+',
             label="$-[p' + b'^2/(2N^2)]w|_{bottom}$")
    plt.plot(t_scl, MET['LDz'][t_strt:], 'g',
             label=r"$-(\nu \partial_z^2 \vec v')\cdot \vec v'$")
    plt.plot(t_scl, MET['HDz'][t_strt:], 'g+',
             label=r"$-(-\nu_4 \partial_z^4 \vec v')\cdot \vec v'$")
    plt.plot(t_scl, MET['LDh'][t_strt:], 'g--',
             label=r"$-(\nu \partial_x^2 \vec v')\cdot \vec v'$")
    plt.plot(t_scl, MET['HDh'][t_strt:], 'g*',
             label=r"$-(-\nu_4 \partial_x^4 \vec v')\cdot \vec v'$")
    plt.plot(t_scl, RES[t_strt:], 'r--', label="Residual")

    plt.grid('on')
    axm = min([MET[ky].min() for ky in MET.keys()])
    listM = [MET[ky].max() for ky in MET.keys()]  # list of maxima
    axM = sorted(listM)[-2]  # assume time is always the largest: 2nd biggest
    plt.axis([t_scl[0], 12., axm, axM])
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
    # base = '/Volumes/LaCie2TB/mbf/'
    base = '/Users/NicoG/FS_Morgan/outputs/'
elif getpass.getuser() == 'nicolas':
    base = '/data/nicolas/mbf/'
elif getpass.getuser() == 'mbfox':
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the base directory?')

plt.close('all')
# boundaries_list = [850, 713, 280, 0]  # right, left, bottom boundaries, start
boundaries_list = [450, 350, 192, 120]
# right, left, bottom boundaries, start, stop

save = 1  # 1 to compute and save, 0 to just load

plt.close('all')

for ii in ['01', '02']:
    for iN in ['50']:
        for iF in ['14']:  # , '12', '14']:
            for iR in ['14']:
                setup = 'N' + iN + 'F' + iF + 'R' + iR + '-loRes-' + ii
                root_dir = base + setup

                print(root_dir)

                pkl_file = open(root_dir+'/spec_params.pkl', 'rb')
                d2pkl = pickle.load(pkl_file)
                pkl_file.close()

                s = dictionary_to_class(**d2pkl)
                del d2pkl

                if save:
                    MET = MEBTerms(s, root_dir, boundaries_list)
                else:
                    MET = np.load(root_dir+'/MEBterms.npz')

                dic_of_starts = {
                    setup[:-3]: 0
                    }

                plot_one_budget(MET, s, setup[:-3], dic_of_starts[setup[:-3]])
