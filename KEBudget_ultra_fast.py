import sys
import matplotlib.pyplot as plt
import numpy as np
# from numpy import shape
from scipy.io import netcdf
import time
import getpass
if (sys.version_info > (3, 0)):  # Then Python 3 is running
    import ngobfftPy3 as tf
else:  # Then is has to be Python 2
    import ngobfft as tf

# %% NG start


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
    FFF = tf.obfft(x, uVar, 2)  # NG correct
    dnudxn = np.real(tf.obifft(k, FFF*(1j*K)**n, 2))
    return uVar*dnudxn


def print_elpsd(str_step, tic):
    print('{0} completed, elapsed time = {1:4.0f} s'.format(str_step,
          time.time() - tic))
    return time.time()


def print_int_elpsd(str_step, counter, ncounters, tic):
    print('    {0}: {0: 3d}% done, {1: 4.0f} s elapsed'.format(
          (100*counter)//ncounters, time.time() - tic))
    counter += 1
    return counter

tic_strt = time.time()

if getpass.getuser() == 'NicoG':
    base = '/Users/NicoG/FS_in-house/outputs/'
elif getpass.getuser() == 'nicolas':
    base = '/data/nicolas/mbf/'
elif getpass.getuser() == 'mbfox':
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the base directory?')

nx_FS = 1024
nz_FS = 512
p1 = 850  # don't know what this is yet
p2 = 713  # idem
p3 = 280  # ibidem

# Below I repatriated stuff that were defined throughout the text

save = 1  # 1 to compute and save, 0 to just load
# NG: I don't think the save feature is used properly though

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

# NG end
# %%

plt.close('all')

#
#
# Importing Data
#
#
setup = base + 'B_Nz_06/'
XZfile = setup + '2D/XZ.nc'
derivsfile = setup + '2D/derivs.nc'
t_derivsfile = setup + '2D/t_derivs.nc'

tic = time.time()

print('Start loading.')

# Opening the Netcdf file
openXZ = netcdf.netcdf_file(XZfile, 'r')

# NG: sometimes we don't want to load everything, simpler to do it one-by-one
# I also want to keep it in a dictionary for easier manipulation in loops
Var = {}
Var['u'] = openXZ.variables['uVar'][:].copy()
Var['v'] = openXZ.variables['vVar'][:].copy()
Var['w'] = openXZ.variables['wVar'][:].copy()
Var['s1'] = openXZ.variables['s1Var'][:].copy()

# %% Opening the Netcdf file
# NG: I will only load pVar for now, because it will be used a lot. I refrain
# from loading the other fields since they're each only used once
openderivs = netcdf.netcdf_file(derivsfile, 'r')

Var['p'] = openderivs.variables['pVar'][:].copy()

opent_derivs = netcdf.netcdf_file(t_derivsfile, 'r')

# # With Python3, one has to convert the dict_keys into a list
# derivsVarList = list(openderivs.variables.keys())
#
# print(openderivs.variables[derivsVarList[1]])

# Converting netcdf to dictionary with variable names as the key
# strpData = {}
# for i in range(0, len(derivsVarList)):
#     temp = openderivs.variables[derivsVarList[i]]
#     varData = temp[:].copy()
#     strpData[derivsVarList[i]] = varData

# Var['p'] = strpData['pVar']  # Pressure data (401, 513, 1024)
# d2udz2 = strpData['d2udz2']
# d4udz4 = strpData['d4udz4']
# d2vdz2 = strpData['d2vdz2']
# d4vdz4 = strpData['d4vdz4']
# d2wdz2 = strpData['d2wdz2']
# d4wdz4 = strpData['d4wdz4']
# d2s1dz2 = strpData['d2s1dz2']
# d4s1dz4 = strpData['d4s1dz4']

# openderivs.close()

print('Loading Vars completed, elapsed time = {0:4.0f} s'.format(
      time.time() - tic))
tic = time.time()

# %% Opening the Netcdf file
"""opent_derivs = netcdf.netcdf_file(t_derivsfile, 'r')

# With Python3, one has to convert the dict_keys into a list
t_derivsVarList = list(opent_derivs.variables.keys())

print(opent_derivs.variables[t_derivsVarList[1]])

# Converting netcdf to dictionary with variable names as the key
strpData = {}
for i in range(0, len(t_derivsVarList)):
    temp = opent_derivs.variables[t_derivsVarList[i]]
    varData = temp[:].copy()
    strpData[t_derivsVarList[i]] = varData

dudt = strpData['dudt']    # U time derivative data (401, 513, 1024)
dvdt = strpData['dvdt']    # V time derivative data
dwdt = strpData['dwdt']    # W time derivative data
# ds1dt = strpData['ds1dt']  # s1 time derivative data

# opent_derivs.close()

print('Loading tderivs completed, elapsed time = {0:4.0f}'.format(
      time.time() - tic))
tic = time.time()"""

# %% NG: I'm not sure if this is in the right place
# Saving
# if not save:
#    PHI = np.load("PHI.npy")
#    EFtot = np.load("EFtot.npy")
#    dEdt = np.load("dEdt.npy")
#    Dtot = np.load("Dtot.npy")
#    Dtot = np.load("Ctot.npy")
#    Ebudget = np.load("Ebudget.npy")

# %%x
t = np.zeros(401)
t = np.linspace(0, 400*dt, 401)  # NG: faster this way
# for k in range(401):
#     t[k] = k*dt

x = np.linspace(0, Lx, nx_FS)
z = np.linspace(-Lz, 0, nz_FS+1)

# Define a box (the CV) around the unstable front
nx = range(p2, p1+1)
nz = range(p3, nz_FS+1)

# %% Compute quantities relative to the basic flow

x0 = x - Lx + xctr0
zshft = z - Lz
X, Zshft = np.meshgrid(x, zshft)
X, Z = np.meshgrid(x, z)
X0, Z = np.meshgrid(x0, z)

alpha = dltH*Lz*(1. - np.exp(-totdepth/(dltH*Lz)))
Bhat = 8.*(3.0**(-1.5))*alpha*(FrG**2)*N2_0/RoG
chi0 = (0.5*Bhat)/(f*np.sqrt(N2_0)*FrG)

Gamma0 = 0.5*(1. - np.tanh(X0/chi0))
dGamma0dx = -(1. + np.tanh(X0/chi0))*(Gamma0/chi0)
d3Gamma0dx3 = (1 - 2*np.sinh(X0/chi0)**2)/(chi0**3 * np.cosh(X0/chi0)**4)

Gamma1 = -np.cosh(X/chi1)**-2
dGamma1dx = -2.*np.tanh(X/chi1)*(Gamma1/chi1)
d3Gamma1dx3 = 8*(np.cosh(X0/chi1)**2 - 3
                 )*np.sinh(X0/chi1)/(chi1**3 * np.cosh(X0/chi1)**5)

Gamma = Gamma0 + Gamma1
dGammadx = dGamma0dx + dGamma1dx
d3Gammadx3 = d3Gamma0dx3 + d3Gamma1dx3

# M^2 is the lateral density frequency
M2 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))

# Background velocity profile (513,1024)
ThW = Bhat * dGammadx * dltH*Lz*(np.exp(Zshft/(dltH*Lz)) -
                                 np.exp(-totdepth/(dltH*Lz)))/f

d2ThWdx2 = Bhat * d3Gammadx3 * dltH*Lz*(np.exp(Zshft/(dltH*Lz)) -
                                        np.exp(-totdepth/(dltH*Lz)))/f

d2ThWdz2 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*dltH*Lz)

# N^2 is the vertical buyoancy frequency
N2_1 = N2_0 / (1 - Zshft / z0Gill) ** 2
N2 = N2_1 + Bhat * Gamma * np.exp(Zshft / dltH / Lz) / dltH / Lz
BN1 = N2_0 * Zshft / (1 - Zshft / z0Gill)
s1ref = -rho0 * (Bhat * np.exp(Zshft / dltH / Lz) * Gamma + BN1) / g

tic = print_elpsd('Geostrophic calculations', tic)

# %%

#
#
# PE calculation
#
#

# calculate buoyancy forces
Var['b'] = -g*(Var['s1'][:, :, :] - s1ref[:, :])/rho0  # NG correct#

# NG added conversion KE to PE

conv = + Var['b']*Var['w']

Ctot = integrate_vol(x[p2:p1+1], z[p3:], conv[:, p3:, p2:p1+1])
del conv
# NG: performing the operation in a funtion has the huge advantage that (1) it
# is perfectly reproducible, (2) less code lines, (3) once the execution is
# done, the intermediate values are erased from memory

# PE_xplus = (b[:, :, p1]*b[:, :, p1])/N2[:, p1]*0.5  # NG multiplied by 0.5
# PE_xminus = (b[:, :, p2]*b[:, :, p2])/N2[:, p2]*0.5  # NG multiplied by 0.5
# PE_zminus = (b[:, p3, :]*b[:, p3, :])/N2[p3, :]*0.5  # NG multiplied by 0.5

tic = print_elpsd('Conversion term', tic)

# %%
#
#
# KE Calculation
#
#

# Calculating KE(per unit volume) at each grid point of interest for use in KE
# calculation
KE_xplus = (Var['u'][:, :, p1]*Var['u'][:, :, p1] +
            Var['v'][:, :, p1]*Var['v'][:, :, p1] +
            Var['w'][:, :, p1]*Var['w'][:, :, p1])*0.5
KE_xminus = (Var['u'][:, :, p2]*Var['u'][:, :, p2] +
             Var['v'][:, :, p2]*Var['v'][:, :, p2] +
             Var['w'][:, :, p2]*Var['w'][:, :, p2])*0.5
KE_zminus = (Var['u'][:, p3, :]*Var['u'][:, p3, :] +
             Var['v'][:, p3, :]*Var['v'][:, p3, :] +
             Var['w'][:, p3, :]*Var['w'][:, p3, :])*0.5

# Assumption is that there is no velocity out of top of CV
# %%

#
#
# Flux calculations
#
#

# Fluxes in x-direction
# phi_xplus = (Var['p'][:, :, p1] + KE_xplus[:, :] + PE_xplus[:, :]
#              )*Var['u'][:, :, p1]  # [kg/s^3]or[W/m^2]
# phi_xminus = -(Var['p'][:, :, p2] + KE_xminus[:, :] + PE_xminus[:, :]
#                )*Var['u'][:, :, p2]
# NG: removed flux of PE
phi_xplus = (Var['p'][:, :, p1] +
             KE_xplus[:, :])*Var['u'][:, :, p1]  # [kg/s^3]or[W/m^2]
del KE_xplus  # NG: makes it easier on memory

phi_xminus = -(Var['p'][:, :, p2] + KE_xminus[:, :])*Var['u'][:, :, p2]
del KE_xminus  # NG makes it easier on memory

# Flux in z-direction
# phi_zminus = -(Var['p'][:, p3, :] + KE_zminus[:, :] + PE_zminus[:, :])*(
#    Var['w'][:, p3, :])
# NG: removed flux of PE
phi_zminus = -(Var['p'][:, p3, :] + KE_zminus[:, :])*Var['w'][:, p3, :]
# Zero flux occurs through top surface
del KE_zminus  # NG: makes it easier on memory

# integrate and add the three paths of energy flux (taking CW path)

PHI1 = np.trapz(phi_xplus[:, p3:], z[p3:], axis=1)  # NG: loops are evil
PHI2 = np.trapz(phi_xminus[:, p3:], z[p3:], axis=1)
PHI3 = np.trapz(phi_zminus[:, p2:p1+1], x[p2:p1+1], axis=1)
PHI = -PHI1 + PHI2 - PHI3

tic = print_elpsd('Fluxes', tic)


# %%

#
#
# Calculation for extracted energy from front
#
#

EF = 0.*Var['p']
EF[:, :, :] = M2[:, :]*(Var['v'][:, :, :]*Var['w'][:, :, :]/f)  # +
#                         Var['u'][:, :, :]*b[:, :, :]/N2[:, :])
# NG just removed the contribution from PE

EFtot = integrate_vol(x[p2:p1+1], z[p3:], EF[:, p3:, p2:p1+1])
del EF

tic = print_elpsd('Conversion term', tic)

# %% NG removed PE

#
#
# Change in KE and NOT PE with time
#
#

# Calculating dKE/dt at every grid point within CV grid
dtKE = 0.*Var['u']
counter = 1

for ii in ['u', 'v', 'w']:
    dtKE += add_deriv_load(opent_derivs, Var[ii], 'd{}dt'.format(ii))
    counter = print_int_elpsd('dKEdt', counter, 3, tic)

dtKEtot = integrate_vol(x[p2:p1+1], z[p3:], dtKE[:, p3:, p2:p1+1])
del dtKE

# NG NOT Calculating dPE/dt at every grid point within CV grid
# dtPE = np.zeros(Var['p'][:, :, :].shape)
# dtPE[:, :, :] = (-g/(rho0*N2[:, :]))*b[:, :, :]*ds1dt[:, :, :]

# NG NOT Performing volume integral to get dPE/dt total for each time step
# dtPEa = np.zeros(Var['p'][:, 1, :].shape)
# dtPEtot = np.zeros(401)
# for i in range(len(Var['p'][:, 1, 1])):
#    for k in range(nx[0], nx[len(nx)-1]+1):
#        dtPEa[i, k] = np.trapz(dtPE[i, p3:, k], z[p3:])
# for i in range(len(Var['p'][:, 1, 1])):
#    dtPEtot[i] = np.trapz(dtPEa[i, p2:p1+1], x[p2:p1+1])

dEdt = dtKEtot  # +dtPEtot

tic = print_elpsd('dEdt', tic)
# %%

#
#
# Compute dissipation terms
#
#

k = tf.k_of_x(x)
K, M = np.meshgrid(k, z)
dffs = {'nuh2': nuh, 'nuh4': -nuh2, 'nuz2': nuz, 'nuz4': -nuz2}
dissip = 0.*Var['u']
counter = 1
tic_tmp = time.time()

print('Start Fourier')
for ii in ['u', 'v', 'w']:
    """ First compute and add each term. Once done, integrate the result.
    Looks like more loops than necessary, but I find it easier to read. """
    for jj in ['2', '4']:
        dzNm = 'd' + jj + ii + 'dz' + jj
        dissip += dffs['nuh'+jj] * add_deriv_load(openderivs, Var[ii], dzNm)
        dissip += dffs['nuz'+jj] * add_deriv_four(x, Var[ii], K, int(jj))
        counter = print_int_elpsd('Dissipation', counter, 6, tic)

# dissipation of the thermal wind
dissip = dissip[:, :, :] + (nuz*d2ThWdz2[:, :] + nuh*d2ThWdx2[:, :]
                            )*Var['v'][:, :, :]

Dtot = integrate_vol(x[p2:p1+1], z[p3:], dissip[:, p3:, p2:p1+1])
del dissip

tic = print_elpsd('Dissipation', tic)

# d2vdz2 = d2vdz2[:, :, :] + d2ThWdz2[:, :]
# del d2ThWdz2
# NG: no hyperviscous operator on the thermal wind (because I wrote is so)
# d4vdz4 = d4vdz4[:, :, :] + d4ThWdz4[:, :]
# d2wdz2 = d2wdz2[:, :, :] + d2ThWdz2[:, :]
# d4wdz4 = d4wdz4[:, :, :] + d4ThWdz4[:, :]

# udisp1 = uVar[:, :, :]*(nuh1*d2udx2[:, :, :] + nuz1*d2udz2[:, :, :] -
#                         nuh2*d4udx4[:, :, :] - nuz2*d4udz4[:, :, :])
# del uVar  # NG makes it easier on memory; that was the last uVar
# del d2udx2
# del d4udx4
# del d2udz2
# del d4udz4

# vdisp1 = Var['v'][:, :, :]*(nuh1*d2vdx2[:, :, :] + nuz1*d2vdz2[:, :, :] -
#                         nuh2*d4vdx4[:, :, :] - nuz2*d4vdz4[:, :, :])

# FFF = tf.obfft(x, uVar, 2)  # NG correct
# DFF = FFF*(1j*K)**2
# d2udx2 = np.real(tf.obifft(k, DFF, 2))
# DFF = FFF*(1j*K)**4
# d4udx4 = np.real(tf.obifft(k, DFF, 2))
# print('Done Dissipation 33%')
# FFF = tf.obfft(x, vVar1, 2)
# del vVar1
# DFF = FFF*(1j*K)**2
# d2vdx2 = np.real(tf.obifft(k, DFF, 2))
# DFF = FFF*(1j*K)**4
# d4vdx4 = np.real(tf.obifft(k, DFF, 2))
# print('Done Dissipation  67%')
# FFF = tf.obfft(x, Var['w'], 2)  # NG correct
# DFF = FFF*(1j*K)**2
# d2wdx2 = np.real(tf.obifft(k, DFF, 2))
# DFF = FFF*(1j*K)**4
# d4wdx4 = np.real(tf.obifft(k, DFF, 2))
# print('Done Dissipation 100%')
# FFF = tf.obfft(x, bVar1, 2)  # NG correct
# DFF = FFF*(1j*K)**2
# d2s1dx2 = np.real(tf.obifft(k, DFF, 2))
# DFF = FFF*(1j*K)**4
# d4s1dx4 = np.real(tf.obifft(k, DFF, 2))
# print('Done Fourier 100%')
# end NG correct

# %%

#
#
# Energy dissipation calculations
#
#

# # Background velocity profile (513, 1024)
# d2ThWdz2 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*dltH*Lz)
# # d4ThWdz4 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*(dltH**3)*(Lz**3))
#
##  Add background profile to fluctutations
## d2udz2 = d2udz2[:, :, :] + d2ThWdz2[:, :]
## d4udz4 = d4udz4[:, :, :] + d4ThWdz4[:, :]
#d2vdz2 = d2vdz2[:, :, :] + d2ThWdz2[:, :]
#del d2ThWdz2
## NG: no hyperviscous operator on the thermal wind (because I wrote is so)
## d4vdz4 = d4vdz4[:, :, :] + d4ThWdz4[:, :]
## d2wdz2 = d2wdz2[:, :, :] + d2ThWdz2[:, :]
## d4wdz4 = d4wdz4[:, :, :] + d4ThWdz4[:, :]
#
#udisp1 = uVar[:, :, :]*(nuh1*d2udx2[:, :, :] + nuz1*d2udz2[:, :, :] -
#                        nuh2*d4udx4[:, :, :] - nuz2*d4udz4[:, :, :])
#del uVar  # NG makes it easier on memory; that was the last uVar
#del d2udx2
#del d4udx4
#del d2udz2
#del d4udz4
#
#vdisp1 = vVar[:, :, :]*(nuh1*d2vdx2[:, :, :] + nuz1*d2vdz2[:, :, :] -
#                        nuh2*d4vdx4[:, :, :] - nuz2*d4vdz4[:, :, :])
#del vVar  # NG makes it easier on memory; that was the last vVar
#del d2vdx2
#del d4vdx4
#del d2vdz2
#del d4vdz4
#
#wdisp1 = Var['w'][:, :, :]*(nuh1*d2wdx2[:, :, :] + nuz1*d2wdz2[:, :, :] -
#                        nuh2*d4wdx4[:, :, :] - nuz2*d4wdz4[:, :, :])
#del Var['w']  # NG makes it easier on memory; that was the last vVar
#del d2wdx2
#del d4wdx4
#del d2wdz2
#del d4wdz4

# s1disp1 = (nuh1*d2s1dx2[:, :, :] + nuz1*d2s1dz2[:, :, :] -
#            nuh2*d4s1dx4[:, :, :] - nuz2*d4s1dz4[:, :, :])*(
#            -g*b[:, :, :]/rho0/N2[:, :])  # that's wrong I think

#print('Start Disp')
#
## Performing surface integrals to get total dissipation at each time step
#udisp1a = np.zeros(Var['p'][:, 1, :].shape)
#udisp1tot = np.zeros(401)
## for i in range(len(Var['p'][:, 1, 1])):
##     for k in range(nx[0], nx[len(nx)-1]+1):
##         udisp1a[i, k] = np.trapz(udisp1[i, p3:, k], z[p3:])
## NG: loops are evil
#udisp1a[i, k] = np.trapz(udisp1[i, p3:, k], z[p3:], axis=1)
#
#del udisp1  # NG
#
#for i in range(len(Var['p'][:, 1, 1])):
#    udisp1tot[i] = np.trapz(udisp1a[i, p2:p1+1], x[p2:p1+1])
#
#del udisp1a  # NG
#print('Done Disp  33%')  # NG correct
#
#vdisp1a = np.zeros(Var['p'][:, 1, :].shape)
#vdisp1tot = np.zeros(401)
#for i in range(len(Var['p'][:, 1, 1])):
#    for k in range(nx[0], nx[len(nx)-1]+1):
#        vdisp1a[i, k] = np.trapz(vdisp1[i, p3:, k], z[p3:])
#
#del vdisp1  # NG
#
#for i in range(len(Var['p'][:, 1, 1])):
#    vdisp1tot[i] = np.trapz(vdisp1a[i, p2:p1+1], x[p2:p1+1])
#
#del vdisp1a  # NG
#print('Done Disp  67%')  # NG correct
#
#wdisp1a = np.zeros(Var['p'][:, 1, :].shape)
#wdisp1tot = np.zeros(401)
#for i in range(len(Var['p'][:, 1, 1])):
#    for k in range(nx[0], nx[len(nx)-1]+1):
#        wdisp1a[i, k] = np.trapz(wdisp1[i, p3:, k], z[p3:])
#
#del wdisp1  # NG
#
#for i in range(len(Var['p'][:, 1, 1])):
#    wdisp1tot[i] = np.trapz(wdisp1a[i, p2:p1+1], x[p2:p1+1])
#
#del wdisp1a  # NG
#print('Done Disp 100%')  # NG correct
#
## NG removed
## s1disp1a = np.zeros(Var['p'][:, 1, :].shape)
## s1disp1tot = np.zeros(401)
## for i in range(len(Var['p'][:, 1, 1])):
##     for k in range(nx[0], nx[len(nx)-1]+1):
##         s1disp1a[i, k] = np.trapz(s1disp1[i, p3:, k], z[p3:])
## for i in range(len(Var['p'][:, 1, 1])):
##     s1disp1tot[i] = np.trapz(s1disp1a[i, p2:p1+1], x[p2:p1+1])
#
#del Var['p']
#
#Dtot = udisp1tot + vdisp1tot + wdisp1tot  # +s1disp1tot
#
#del udisp1tot  # NG
#del vdisp1tot  # NG
#del wdisp1tot  # NG

# print('Done Disp 100%')
# %%

#
#
# Full accounting of Energy Budget -> checking for LHS=RHS and all energy is
# conserved
#
#

fig1 = plt.figure()
plt.plot(t, -PHI, 'k', label='Flux')
# plt.xlabel("Time [$s$]")
# plt.ylabel("Integrated Energy Flux [$W$]")
# plt.title("Energy Flux Across Boundary Around Unstable Front")

# fig2 = plt.figure()
plt.plot(t, -EFtot, 'm', label='Extracted')
# plt.xlabel("Time [$s$]")
# plt.ylabel("Integrated Energy Extracted from Front [$W$]")
# plt.title("Energy Extracted from Unstable Front")

# fig3 = plt.figure()
plt.plot(t, -dEdt, 'b', label='dEdt')
# plt.xlabel("Time [$s$]")
# plt.ylabel("$d(KE+PE)/dt$ [$W$]")
# plt.title("Change in Total Energy With Time")

# fig4 = plt.figure()
plt.plot(t, Dtot, 'g', label='Dissipation')
# plt.xlabel("Time [$s$]")
# plt.ylabel("Energy Dissipation [$W$]")
# plt.title("Energy Dissipation Within Front Over Time")

# NG added
plt.plot(t, Ctot, 'c', label='Conversion')

Ebudget = -PHI - EFtot - dEdt + Dtot - 0*Ctot
# fig5 = plt.figure()
plt.plot(t, Ebudget, 'r', label='Cumulative Budget', lw=2)

plt.xlabel("Time [$s$]")
plt.ylabel("Integrated Energy Per Unit Density [$m^4/s^3$]")
plt.title("Full Accounting of Energy Budget")
plt.legend()
plt.show()


# %%
# Saving and concluding
if save == 1:
    np.save(setup+"PHI.npy", PHI)
    np.save(setup+"EFtot.npy", EFtot)
    np.save(setup+"dEdt.npy", dEdt)
    np.save(setup+"Dtot.npy", Dtot)
    np.save(setup+"Ctot.npy", Ctot)
    np.save(setup+"Ebudget.npy", Ebudget)

openXZ.close()
openderivs.close()
opent_derivs.close()

time_tot = time.time() - tic_strt
mins = int(time_tot/60)
secs = int(time_tot - 60*mins)

print(' ')
print('          ******')
print(' ')
print('All done! Elapsed time = {0:3d}:{0:2d}'.format(mins, secs))
print(' ')
print('          ******')
print(' ')
