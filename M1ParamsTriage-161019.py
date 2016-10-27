
import os
import pickle
import numpy as np
from scipy.io import netcdf as nc
# import matplotlib.pyplot as plt
# import sys
import getpass
import sys
if (sys.version_info > (3, 0)):  # Then Python 3 is running
    sys.path.append('/Users/NicoG/Dropbox/python_useful/obngfft')
    import ngobfftPy3 as tf
else:  # Then is has to be Python 2
    sys.path.append('/Users/NicoG/Dropbox/python_useful/obngfft')
    import ngobfft as tf


def replace_d_by_e(strng):
    ListOfChars = list(strng)
    if ListOfChars.count('d') > 0:
        ListOfChars[ListOfChars.index('d')] = 'e'
    return "".join(ListOfChars)


def read_one_set(root, fl):
    print(root)
    ProbParPath = '{0}codes_etc/input/problem_params'.format(root)

    if not os.path.exists(ProbParPath):
        print('    => does not exist')
        return

    # %% ----------------------------------------------------------------------
    # problem_params is relatively stable in terms of what is where.
    # Therefore, it is possible to assume that the line numbers won't change

    fid = open(ProbParPath)
    lines = fid.readlines()
    fid.close()

    d2pkl = {
        'nx': int(lines[1][:15]),
        'ny': int(lines[2][:15]),
        'nz': int(lines[3][:15]),
        'nL': int(lines[4][:15]),
        'nsp': int(lines[5][:15]),
        'dt': float(replace_d_by_e(lines[6][:15])),
        't_end': float(replace_d_by_e(lines[8][:15])),
        'BCs': lines[9][:15].strip(),
        'Lx': float(replace_d_by_e(lines[10][:15])),
        'Ly': float(replace_d_by_e(lines[11][:15])),
        'Lz': float(replace_d_by_e(lines[12][:15])),
        'g': float(replace_d_by_e(lines[16][:15])),
        'f': float(replace_d_by_e(lines[17][:15])),
        'rho_0': float(replace_d_by_e(lines[18][:15])),
        'nuh': float(replace_d_by_e(lines[19][:15])),
        'nuz': float(replace_d_by_e(lines[20][:15])),
        'kappah': float(replace_d_by_e(lines[21][:15])),
        'kappaz': float(replace_d_by_e(lines[22][:15])),
        'hdeg': int(lines[25][:15]),
        'nuh2': float(replace_d_by_e(lines[30][:15])),
        'nuz2': float(replace_d_by_e(lines[31][:15])),
        'kappah2': float(replace_d_by_e(lines[32][:15])),
        'kappaz2': float(replace_d_by_e(lines[33][:15])),
        'hdeg2': int(lines[36][:15]),
        'nlflag': int(lines[37][:15])}

    # %% ----------------------------------------------------------------------
    # Contrary to problem_params, the user_params module in user_module.f90
    # changes often. Better use a more flexible approach.

    UsrModPath = '{0}codes_etc/input/user_module.f90'.format(root)
    fid = open(UsrModPath)
    # lines = fid.readlines()

    # um = {} # user_module dictionary
    for line in fid:
        els = line.split()
        if line.count(':: '):
            idx_eq = els.index('=')
            var = els[idx_eq-1]
            val = els[idx_eq+1]
            if els[0].count('real(kind=8)'):
                d2pkl[var] = float(replace_d_by_e(val))
            elif els[0].count('integer'):
                d2pkl[var] = int(val)
            elif els[0].count('logical'):
                if val.lower().count('true'):
                    d2pkl[var] = True
                else:
                    d2pkl[var] = False
        elif line.count('end module user_params'):
            break

    fid.close()

    # %% ----------------------------------------------------------------------
    # Reading io_params

    fid = open('{0}codes_etc/input/io_params'.format(root))
    lines = fid.readlines()
    fid.close()

    idx = 0
    ln_cnt = 0
    line = lines[0]
    while line[:line.index('!')].strip() != fl:
        ln_cnt += 1
        line = lines[ln_cnt]

    idx = ln_cnt
    d2pkl.update({
        'RecType': lines[idx+1][:lines[idx+1].index('!')].strip(),
        'sk_it': int(lines[idx+2][:lines[idx+2].index('!')])})

    if d2pkl['RecType'] == 'new':
        d2pkl['iteID'] = '_0000000000'
    elif d2pkl['RecType'] == 'append':
        d2pkl['iteID'] = ''
    else:
        raise NameError('No RecType? io_params not read.')

    # Number of procs
    ListFl = os.listdir('{0}2D/'.format(root))
    ListFl_tmp = os.listdir('{0}2D/'.format(root))
    # print(ListFl_tmp)
    for item in ListFl:
        # print(item)
        # print(item[:len(fl)])
        # print(fl)
        if item[:len(fl)] != fl:
            ListFl_tmp.remove(item)
            # print(ListFl_tmp)

    d2pkl['np'] = len(
        [nm for nm in ListFl_tmp if nm.count('{0}'.format(d2pkl['iteID']))])

    # print('Number of processors = {}'.format(d2pkl['np']))

    if d2pkl['np'] == 1:
        d2pkl['npID'] = ''
    elif d2pkl['np'] > 1:
        d2pkl['npID'] = '_000'

    # Number of iterations
    d2pkl['nites'] = int(np.floor(d2pkl['t_end']/d2pkl['dt'])) + 1

    # %% ----------------------------------------------------------------------
    # Coordinates

    rbf = nc.netcdf_file('{0}2D/{1}{2}{3}.nc'.format(root, fl, d2pkl['iteID'],
                         d2pkl['npID']), 'r')
    d2pkl.update({
        'x': rbf.variables['xVar'][:].copy(),
        'z': rbf.variables['zVar'][:].copy(),
        't': rbf.variables['tVar'][:].copy(),
        'V': d2pkl['Lx']*d2pkl['Lz'],
        'dx': d2pkl['Lx']/d2pkl['nx'],
        'dy': d2pkl['Ly']/d2pkl['ny'],
        'dz': d2pkl['Lz']/d2pkl['nz']})
    if d2pkl['ny'] > 1:
        d2pkl['y'] = rbf.variables['yVar'][:].copy()
        d2pkl['V'] *= d2pkl['Ly']
    rbf.close()

    d2pkl['xs'] = d2pkl['x'] - d2pkl['Lx']*0.5

    # vertical coordinate
    if d2pkl['np'] > 1:
        for ii in range(d2pkl['np'])[1:]:
            pID = '_{0:03d}'.format(ii)
            rbf = nc.netcdf_file('{0}2D/{1}{2}.nc'.format(
                root, 'rhobar_0000000000', pID), 'r')
            d2pkl['z'] = np.concatenate(
                (d2pkl['z'], rbf.variables['zVar'][:].copy()))
            rbf.close()

    d2pkl['zs'] = d2pkl['z'] - d2pkl['Lz']

    d2pkl['nnz'] = len(d2pkl['z'])

    # grids
    d2pkl['xz'], d2pkl['zx'] = np.meshgrid(d2pkl['x'], d2pkl['z'])
    if d2pkl['ny'] > 1:
        d2pkl['xy'], d2pkl['yx'] = np.meshgrid(d2pkl['x'], d2pkl['y'])
        d2pkl['yz'], d2pkl['zy'] = np.meshgrid(d2pkl['y'], d2pkl['z'])

    d2pkl['zxs'] = d2pkl['zx'] - d2pkl['Lz']
    d2pkl['xzs'] = d2pkl['xz'] - d2pkl['xctr0']
    d2pkl['xz0'] = d2pkl['xz'] - d2pkl['Lx'] + d2pkl['xctr0']

    # print('Lz = {}'.format(d2pkl['Lz']))

    # time coordinate
    if d2pkl['RecType'] == 'new':
        for ii in range(d2pkl['nites'])[1:]:
            pID = '_{0:010d}'.format(ii)
            rbf = nc.netcdf_file(
                '{0}2D/{1}{2}{3}.nc'.format(root, fl, pID, d2pkl['npID']), 'r')
            d2pkl['t'] = np.concatenate(
                (d2pkl['t'], rbf.variables['tVar'][:].copy()))
            rbf.close()

    # spectral coordinates
    d2pkl['k'] = tf.k_of_x(d2pkl['x'])
    d2pkl['dk'] = d2pkl['k'][1] - d2pkl['k'][0]

    if d2pkl['BCs'] == 'zslip':
        d2pkl['m'] = tf.k_of_x(d2pkl['z'][:-1])
    elif d2pkl['BCs'] == 'zperiodic':
        d2pkl['m'] = tf.k_of_x(d2pkl['z'])
    d2pkl['dm'] = d2pkl['m'][1] - d2pkl['m'][0]

    d2pkl['km'], d2pkl['mk'] = np.meshgrid(d2pkl['k'], d2pkl['m'])

    if d2pkl['ny'] > 1:
        d2pkl['l'] = tf.k_of_x(d2pkl['y'])
        d2pkl['dl'] = d2pkl['l'][1] - d2pkl['l'][0]

        d2pkl['kl'], d2pkl['lk'] = np.meshgrid(d2pkl['k'], d2pkl['l'])
        d2pkl['lm'], d2pkl['ml'] = np.meshgrid(d2pkl['l'], d2pkl['m'])

    # Misc:
    Omega = 2.*np.pi/86400
    Rearth = 6375.e3
    lat = np.arcsin(0.5*d2pkl['f']/Omega)
    if d2pkl['beta_y_n']:
        d2pkl['beta'] = -2.*Omega*np.cos(lat)/Rearth
    else:
        d2pkl['beta'] = 0.
    d2pkl['f_local'] = d2pkl['f'] + d2pkl['beta']*d2pkl['xzs']
    d2pkl['f0'] = d2pkl['f'] + d2pkl['beta']*(d2pkl['Lx']*0.5 -
                                              d2pkl['xctr0'])

    # %% ----------------------------------------------------------------------
    # Geostrophic flow and stratification
    # Not ready for 3D

    if d2pkl['dltH'] > 0. and d2pkl['FrG'] > 0.:
        alpha1 = d2pkl['dltH']*d2pkl['Lz'] * (
            1. - np.exp(-d2pkl['totdepth']/d2pkl['dltH']/d2pkl['Lz']))
        alpha2 = np.exp(d2pkl['zxs']/d2pkl['Lz']/d2pkl['dltH'])
        alpha3 = np.exp(d2pkl['zxs']/d2pkl['dltH']/d2pkl['Lz']
                        ) / d2pkl['dltH'] / d2pkl['Lz']
        alpha4 = (np.exp(d2pkl['zxs']/d2pkl['dltH']/d2pkl['Lz']) -
                  np.exp(-d2pkl['totdepth']/d2pkl['dltH']/d2pkl['Lz'])
                  ) * d2pkl['dltH']*d2pkl['Lz']
    elif d2pkl['dltH'] <= 0. and d2pkl['FrG'] > 0.:
        alpha1 = d2pkl['totdepth']
        alpha2 = 1.
        alpha3 = 0.
        alpha4 = d2pkl['zxs'] + d2pkl['totdepth']
    else:
        alpha1 = 0.
        alpha2 = 0.
        alpha3 = 0.
        alpha4 = 0.

    Bhat = (8*3.**(-1.5) * alpha1 * d2pkl['FrG']**2 * d2pkl['up_N2'] /
            d2pkl['RoG'])
    if d2pkl['FrG'] > 0.:
        dltS2 = 0.5 * Bhat / (abs(d2pkl['f0']) * np.sqrt(d2pkl['up_N2']) *
                              d2pkl['FrG'])
    else:
        dltS2 = 1.e8

    # Main front:
    Gamma0 = 0.5 * (1. - np.tanh(d2pkl['xz0'] / dltS2))
    dGamma0dx = -(1. + np.tanh(d2pkl['xz0'] / dltS2)) * Gamma0 / dltS2
    d2Gamma0dx2 = 2. * np.tanh(d2pkl['xz0'] / dltS2) * Gamma0 * (
        1. + np.tanh(d2pkl['xz0']/dltS2)) / dltS2**2

    # print('max(x) = {}'.format(d2pkl['xz'].max()))
    # print('max(x0) = {}'.format(d2pkl['xz0'].max()))

    # Secondary front ensuring x-periodicity of N^2
    Gamma1 = -1. / np.cosh(d2pkl['xz']/d2pkl['chi1'])**2
    dGamma1dx = -2. * np.tanh(d2pkl['xz']/d2pkl['chi1']
                              ) * Gamma1 / d2pkl['chi1']
    d2Gamma1dx2 = 2. * Gamma1 * (
        2. - 3./np.cosh(d2pkl['xz']/d2pkl['chi1'])**2) / d2pkl['chi1']**2

    # Total front-induced buoyancy perturbation shape
    Gamma = Gamma0 + Gamma1
    dGammadx = dGamma0dx + dGamma1dx
    d2Gammadx2 = d2Gamma0dx2 + d2Gamma1dx2

    # Non-frontal stratification
    if d2pkl['z0_Gill'] < 0.:
        N21 = d2pkl['up_N2']
        BN1 = d2pkl['up_N2'] * d2pkl['zxs']
        dN21dz = 0.
    else:
        N21 = d2pkl['up_N2'] * (1 - d2pkl['zxs']/d2pkl['z0_Gill'])**(-2)
        BN1 = d2pkl['up_N2'] * d2pkl['zxs'] * (
            1 - d2pkl['zxs']/d2pkl['z0_Gill'])**(-1)
        dN21dz = d2pkl['up_N2'] * 2. * (
            1 - d2pkl['zxs']/d2pkl['z0_Gill'])**(-3) / d2pkl['z0_Gill']

    # Geostrophic Flow dictionary
    d2pkl.update({
        'RLoc': (BN1 + Bhat*Gamma*alpha2) * (-d2pkl['rho_0']/d2pkl['g']),
        'N2Loc':  N21 + Bhat*Gamma*alpha3,
        'S2Loc':  Bhat*dGammadx*alpha2,
        'TWLoc':  Bhat*dGammadx*alpha4 / d2pkl['f_local'],
        'RoGLoc':  Bhat*(d2pkl['f_local']*d2Gammadx2 - d2pkl['beta']*dGammadx
                         )*alpha4 / d2pkl['f_local']**3,
        'd2Rdx2':  Bhat*d2Gammadx2*alpha2 * (-d2pkl['rho_0']/d2pkl['g']),
        'd2Rdz2': (Bhat*Gamma*alpha3/(d2pkl['dltH']*d2pkl['Lz']) + dN21dz
                   ) * (-d2pkl['rho_0']/d2pkl['g'])})

    #    plt.figure()
    #    plt.pcolormesh(d2pkl['xz'], d2pkl['zx'], d2pkl['RLoc'])
    #    plt.colorbar()
    #    plt.show()

    # %% ----------------------------------------------------------------------
    # We can pickle that!

    os.system('rm {0}spec_params.pkl'.format(root))
    pkl_file = open('{0}spec_params.pkl'.format(root), 'wb')
    pickle.dump(d2pkl, pkl_file, -1)
    pkl_file.close()


# %%
if getpass.getuser() == 'NicoG':
    base = '/Users/NicoG/FS_Morgan/outputs/'
    # base = '/Users/NicoG/FS_in-house/outputs/'
    # base = '/Volumes/LaCie2TB/mbf/'
elif getpass.getuser() == 'nicolas':
    base = '/data/nicolas/mbf/'
elif getpass.getuser() == 'mbfox':
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the base directory?')

save_o_n = True
plot_o_n = False

fl = 'XZ'

for ii in ['01']:  # 13:16: # experiment index
    for iN in ['50']:
        for iF in ['14']:  # ['10','12','14']:
            for iR in ['14']:  #['10','12','14']:
                setup = 'N' + iN + 'F' + iF + 'R' + iR + '-loRes-' + ii + '/'
                root = base + setup
                read_one_set(root, fl)

# for setup in ['F10R14', 'F12R10']:
#     root = base + 'N68' + setup + '-01/'
#     read_one_set(root, fl)
