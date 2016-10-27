import sys
import matplotlib.pyplot as plt
import numpy as np
# from numpy import shape
from scipy.io import netcdf
import time
import getpass
import FD
# import scipy.sparse import csr_matrix
if (sys.version_info > (3, 0)):  # Then Python 3 is running
    import ngobfftPy3 as tf
else:  # Then is has to be Python 2
    import ngobfft as tf

# %%


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
    FFF = tf.obfft(x, uVar, 1)
    dnudxn = np.real(tf.obifft(k, FFF*(1j*K)**n, 1))
    return dnudxn


def print_elpsd(str_step, tic):
    print('{0} completed, elapsed time = {1: 4d} s'.format(str_step,
          int(time.time() - tic)))
    return time.time()


def print_int_elpsd(str_step, counter, ncounters, tic):
    print('    {0}: {1: >3d}% done, {2: 4d} s elapsed'.format(str_step,
          int((100*counter)/ncounters), int(time.time() - tic)))
    counter += 1
    return counter


def visco_calc(eqn_id, dID, x, K, ueqn, dict_d):
    """
    eqn_id [string]: 'u', 'v', 'w', 's1'
    dID [open file ID]: the container of pVar, d2udz2, etc
    x [1D nparray]: the x vector
    K [2D nparray]: the replicated k vector
    ueqn [nparray]: u, v, w, or s1'
    dict_diffs [dictionnary]: dictionnary of the diffusivities
    """
    d2udz2 = dID.variables['d2'+eqn_id+'dz2'][tstep, :, :].copy()
    d4udz4 = dID.variables['d4'+eqn_id+'dz4'][tstep, :, :].copy()
    d2udx2 = add_deriv_four(x, ueqn, K, 2)
    d4udx4 = add_deriv_four(x, ueqn, K, 4)

    return (dict_d['nuh']*d2udx2 + dict_d['nuz']*d2udz2 -
            dict_d['nuh2']*d4udx4 - dict_d['nuz2']*d4udz4)


def ithEqnBudget(eqn_id, dtID, dict_terms):
    """ This routine does two things:
    1. loads readily available quantities like dudt
    2. plots the stuff
    eqn_id [string]: 'u', 'v', 'w', 's1'
    dtID [open file ID]: the container of dudt, etc
    dict_Var [dict]: the dictonnary that contains tuples, 1st element being
        name of the term, second the arrays of the terms"""

    # unsteadiness
    dict_terms['dudt'] = ('d'+eqn_id+'dt',
                          dtID.variables['d'+eqn_id+'dt'][tstep, :, :].copy())

    # residual
    dict_terms['epsilon'] = (eqn_id+'epsilon',
                             dict_terms['dudt'][1] + dict_terms['NL'][1] +
                             dict_terms['xtra1'][1] + dict_terms['xtra2'][1] -
                             dict_terms['visco'][1])

    # plot
    plt.figure(eqn_id+' equation')
    i = 0
    for ii in ['dudt', 'NL', 'xtra1', 'xtra2', 'visco', 'epsilon']:
        plt.subplot(231+i)
        i += 1
        valmax = abs(dict_terms[ii][1][:, :]).max()
        plt.imshow(dict_terms[ii][1][:-2, :], clim=(-valmax, valmax))
        plt.title(dict_terms[ii][0])
        plt.colorbar(orientation='horizontal', format='%1.1e')

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    plt.show()


plt.close("all")

tic_strt = time.time()

if getpass.getuser() == 'NicoG':
    base = '/Users/NicoG/FS_in-house/outputs/'
elif getpass.getuser() == 'nicolas':
    base = '/data/'
elif getpass.getuser() == 'mbfox':
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the base directory?')

nx_FS = 1024
nz_FS = 512
p1 = 850  # right boundary
p2 = 713  # left boundary
p3 = 280  # bottom boundary

# Below I repatriated stuff that were defined throughout the text

dt = 4500.  #
Lx = 3.5e4
Lz = 50
rho0 = 1025.  # [kg/m^3]
g = 9.81  # [m/s^2]
N2_0 = 4.9e-5  # Stratification Frequency

f = 1.03e-4
dltH = 1.00
totdepth = 5000.0
z0Gill = 4e2
FrG = 1.20
RoG = 1.20
chi1 = 5.0e3
xctr0 = 75e2

nuh = 1e-5  # NG: I am changing the signs form problem_params
nuh2 = 2e3  # because it's easier to manipulate when computing the dissipation
nuz = 1e-5  # terms
nuz2 = 1e-7

tstep = -1

# NG end
# %%

plt.close('all')

#
#
# Importing Data
#
#
setup = base + 'B_Nz_07/'
XZfile = setup + '2D/XZ.nc'
derivsfile = setup + '2D/derivs.nc'
t_derivsfile = setup + '2D/t_derivs.nc'

tic = time.time()

print('Start loading.')

# Opening the Netcdf files. Opening isn't loading: we can keep them open as
# long as we want, and as long as we close them atthe end.
openXZ = netcdf.netcdf_file(XZfile, 'r')
openderivs = netcdf.netcdf_file(derivsfile, 'r')
opent_derivs = netcdf.netcdf_file(t_derivsfile, 'r')

# NG: sometimes we don't want to load everything, simpler to do it one-by-one
# I also want to keep it in a dictionary for easier manipulation in loops
Var = {}
Var['u'] = openXZ.variables['uVar'][tstep, :, :].copy()
Var['v'] = openXZ.variables['vVar'][tstep, :, :].copy()
Var['w'] = openXZ.variables['wVar'][tstep, :, :].copy()
Var['s1'] = openXZ.variables['s1Var'][tstep, :, :].copy()
Var['p'] = openderivs.variables['pVar'][tstep, :, :].copy()
Var['vx'] = openXZ.variables['vort_xVar'][tstep, :, :].copy()
Var['vy'] = openXZ.variables['vort_yVar'][tstep, :, :].copy()
Var['vz'] = openXZ.variables['vort_zVar'][tstep, :, :].copy()


# %%
t = np.zeros(401)
t = np.linspace(0, 400*dt, 401)  # NG: faster this way

# x = np.linspace(0, Lx, nx_FS)
# z = np.linspace(-Lz, 0, nz_FS+1)
x = openXZ.variables['xVar'][:].copy()  # The linspace thing above isn't
z = openXZ.variables['zVar'][:].copy()  # exactly the actual x...

dz = z[1] - z[0]

# nx = range(p2, p1+1)
# nz = range(p3, nz_FS+1)


# %% Compute quantities relative to the basic flow

x0 = x - Lx + xctr0
zshft = z - Lz
X, Zshft = np.meshgrid(x, zshft)
X0, Z = np.meshgrid(x0, z)

alpha1 = dltH*Lz*(1. - np.exp(-totdepth/dltH/Lz))
alpha2 = np.exp(Zshft/dltH/Lz)
alpha3 = np.exp(Zshft/dltH/Lz) / dltH/Lz
alpha4 = dltH*Lz * (np.exp(Zshft/dltH/Lz) - np.exp(-totdepth/dltH/Lz))

Bhat = 8*3.**(-1.5) * alpha1 * FrG**2 * N2_0 / RoG
chi0 = 0.5 * Bhat / (abs(f)*np.sqrt(N2_0)*FrG)

# ###### Main front:
Gamma0 = 0.5 * (1. - np.tanh(X0/chi0))
dGamma0dx = -(1. + np.tanh(X0/chi0)) * Gamma0 / chi0
d2Gamma0dx2 = 2 * np.tanh(X0/chi0) * Gamma0 * (1. + np.tanh(X0/chi0))/chi0**2
d3Gamma0dx3 = (1 - 2*np.sinh(X0/chi0)**2)/(chi0**3 * np.cosh(X0/chi0)**4)

# ###### Secondary front ensuring x-periodicity of N^2
Gamma1 = -1./np.cosh(X/chi1)**2
dGamma1dx = -2*np.tanh(X/chi1) * Gamma1 / chi1
d2Gamma1dx2 = 2*Gamma1 * (2 - 3./np.cosh(X/chi1)**2) / chi1**2
d3Gamma1dx3 = -4*(np.cosh(X0/chi1)**2 - 3) * dGamma1dx * Gamma1 / chi1**2

# ###### Total front-induced buoyancy perturbation shape
Gamma = Gamma0 + Gamma1
dGammadx = dGamma0dx + dGamma1dx
d2Gammadx2 = d2Gamma0dx2 + d2Gamma1dx2
d3Gammadx3 = d3Gamma0dx3 + d3Gamma1dx3

N21 = N2_0 * (1. - Zshft/z0Gill)**(-2)
BN1 = N2_0 * Zshft * (1 - Zshft/z0Gill)**(-1)
dN21dz = N2_0 * 2 * (1 - Zshft/z0Gill)**(-3) / z0Gill

s1ref = (BN1 + Bhat*Gamma*alpha2) * -rho0/g
N2 = N21 + Bhat * Gamma * alpha3
M2 = Bhat * dGammadx * alpha2
ThW = Bhat * dGammadx * alpha4 / f
dThWdx = Bhat * d2Gammadx2 * alpha4 / f

d2ThWdx2 = Bhat * d3Gammadx3 * alpha4 / f
d2ThWdz2 = M2 / (f*dltH*Lz)

d2Rdx2 = -rho0 * Bhat * d2Gammadx2 * alpha2 / g
d2Rdz2 = (dN21dz + Bhat * Gamma * alpha3 / (dltH*Lz)) * -rho0/g

tic = print_elpsd('Geostrophic calculations', tic)


# %% misc derivatives

k = tf.k_of_x(x)
K, M = np.meshgrid(k, z)
D1 = FD.getFDMatrix(nz_FS+1, dx=dz, accuracy_order=4).toarray()
diffs = {'nuh': nuh, 'nuz': nuz, 'nuh2': nuh2, 'nuz2': nuz2}
s1prime = Var['s1'][:, :] - s1ref[:, :]


# %% test derivative
"""
aa = D1.dot(ThW)
bb = M2/f

plt.figure(1)
plt.subplot(221)
plt.imshow(aa)
plt.colorbar(orientation='horizontal', format='%1.1e')
plt.subplot(222)
plt.imshow(bb)
plt.colorbar(orientation='horizontal', format='%1.1e')
plt.subplot(223)
plt.imshow(bb-aa)
plt.colorbar(orientation='horizontal', format='%1.1e')

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()
"""

# %%

for fld in ['u', 'v', 'w', 's1']:
    dic = {}
    if fld == 's1':
        dudx = add_deriv_four(x, s1prime, K, 1)
        dudz = D1.dot(s1prime)
        dic['NL'] = ['NLs1', Var['u']*dudx + Var['w']*dudz]
    elif fld == 'u':
        dic['NL'] = ['NLx', Var['w']*Var['vy'] - Var['v']*Var['vz']]
    elif fld == 'v':
        dic['NL'] = ['NLy', Var['u']*Var['vz'] - Var['w']*Var['vx']]
    elif fld == 'w':
        dic['NL'] = ['NLz', Var['v']*Var['vx'] - Var['u']*Var['vy']]

    if fld == 'v':
        visco_tmp = visco_calc('v', openderivs, x, K, Var['v'], diffs)
        # vertical derivatives of v in openderivs only inlude fluctuations
        dic['visco'] = [fld+'visco', visco_tmp + nuh*d2ThWdx2 + nuz*d2ThWdz2]
    elif fld == 's1':
        visco_tmp = visco_calc('s1', openderivs, x, K, s1prime, diffs)
        # vertical derivatives of s1 in openderivs only inlude fluctuations
        dic['visco'] = [fld+'visco', visco_tmp + nuh*d2Rdx2 + nuz*d2Rdz2]
    else:
        dic['visco'] = [fld+'visco', visco_calc(fld, openderivs, x, K,
                                                Var[fld], diffs)]

    if fld == 'u':
        dic['xtra2'] = ['dpdx', add_deriv_four(x, Var['p'], K, 1)]
        dic['xtra1'] = ['-fv', -f*Var['v']]
    elif fld == 'v':
        dic['xtra1'] = ('(f+dThWdx)*u', (f+dThWdx)*Var['u'])
        dic['xtra2'] = ('M2*w/f', M2*Var['w']/f)
    elif fld == 'w':
        dic['xtra2'] = ['dpdz', D1.dot(Var['p'])]
        dic['xtra1'] = ['-b', +g*s1prime/rho0]
    elif fld == 's1':
        dic['xtra1'] = ['-rho0*M2*u/g', -rho0*M2*Var['u']/g]
        dic['xtra2'] = ['-rho0*N2*w/g', -rho0*N2*Var['w']/g]

    ithEqnBudget(fld, opent_derivs, dic)

openXZ.close()
openderivs.close()
opent_derivs.close()
