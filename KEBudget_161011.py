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
    # avoiding loops divided the execution time by 13
    uint1 = np.trapz(u, z, axis=1)
    uint2 = np.trapz(uint1, x, axis=1)
    return uint2


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

nuh1 = 1e-5
nuh2 = 2e3
nuz1 = 1e-5
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

# Opening the Netcdf file
openXZ = netcdf.netcdf_file(XZfile, 'r')

# Converting netcdf to dictionary with variable names as the key
strpData = {}
uVar = openXZ.variables['uVar'][:].copy()
vVar = openXZ.variables['vVar'][:].copy()
wVar = openXZ.variables['wVar'][:].copy()
s1Var = openXZ.variables['s1Var'][:].copy()

# XZVarList = list(openXZ.variables.keys())

# With Python3, one has to convert the dict_keys into a list
# print(openXZ.variables[XZVarList[1]])

# for i in range(0, len(XZVarList)):
#    temp = openXZ.variables[XZVarList[i]]
#    varData = temp[:].copy()
#    strpData[XZVarList[i]] = varData

# uVar = strpData['uVar']    # X velocity (401, 513, 1024)
# vVar = strpData['vVar']    # V velocity
# wVar = strpData['wVar']    # Z velocity
# s1Var = strpData['s1Var']  # Density data

print('Loading XZ completed, elapsed time = {}'.format(time.time() - tic))
tic = time.time()

# %% Opening the Netcdf file
openderivs = netcdf.netcdf_file(derivsfile, 'r')

pVar = openderivs.variables['pVar'][:].copy()

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

# pVar = strpData['pVar']  # Pressure data (401, 513, 1024)
# d2udz2 = strpData['d2udz2']
# d4udz4 = strpData['d4udz4']
# d2vdz2 = strpData['d2vdz2']
# d4vdz4 = strpData['d4vdz4']
# d2wdz2 = strpData['d2wdz2']
# d4wdz4 = strpData['d4wdz4']
# d2s1dz2 = strpData['d2s1dz2']
# d4s1dz4 = strpData['d4s1dz4']

# openderivs.close()

print('Loading derivs completed, elapsed time = {}'.format(time.time() - tic))
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

print('Loading tderivs completed, elapsed time = {}'.format(time.time() - tic))
tic = time.time()"""

# %% NG added
del strpData
# NG: note that it would be nice to not load the unused stuff, but I don't have
# the time to do so.

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

alpha = dltH*Lz*(1.0 - np.exp(-totdepth/(dltH*Lz)))
Bhat = 8.0*(3.0**(-1.5))*alpha*(FrG**2)*N2_0/RoG
chi0 = (0.5*Bhat)/(f*np.sqrt(N2_0)*FrG)

Gamma0 = 0.5*(1.0 - np.tanh(X0/chi0))
dGamma0dx = -(1.0 + np.tanh(X0/chi0))*(Gamma0/chi0)

Gamma1 = -1.0/(np.cosh(X/chi1)**2)
dGamma1dx = -2.0*np.tanh(X/chi1)*(Gamma1/chi1)

Gamma = Gamma0 + Gamma1
dGammadx = dGamma0dx + dGamma1dx

# M^2 is the lateral density frequency
M2 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))

# Background velocity profile (513,1024)
ThW = Bhat*dGammadx*dltH*Lz*(np.exp(Zshft/(dltH*Lz)) -
                             np.exp(-totdepth/(dltH*Lz)))/f

# N^2 is the vertical buyoancy frequency
N2_1 = N2_0 / (1 - Zshft / z0Gill) ** 2
N2 = N2_1 + Bhat * Gamma * np.exp(Zshft / dltH / Lz) / dltH / Lz
BN1 = N2_0 * Zshft / (1 - Zshft / z0Gill)
s1ref = -rho0 * (Bhat * np.exp(Zshft / dltH / Lz) * Gamma + BN1) / g

# %%

#
#
# PE calculation
#
#

# calculate buoyancy forces
bVar = -g*(s1Var[:, :, :] - s1ref[:, :])/rho0  # NG correct#

# NG added conversion KE to PE

conv = + bVar*wVar

del bVar

Ctot = integrate_vol(x[p2:p1+1], z[p3:], conv[:, p3:, p2: p1+1:])
# NG: performing the operation in a funtion has the huge advantage that (1) it
# is perfectly reproducible, (2) less code lines, (3) once the execution is
# done, the intermediate values are erased from memory

# PE_xplus = (b[:, :, p1]*b[:, :, p1])/N2[:, p1]*0.5  # NG multiplied by 0.5
# PE_xminus = (b[:, :, p2]*b[:, :, p2])/N2[:, p2]*0.5  # NG multiplied by 0.5
# PE_zminus = (b[:, p3, :]*b[:, p3, :])/N2[p3, :]*0.5  # NG multiplied by 0.5


# %%
#
#
# KE Calculation
#
#

# Calculating KE(per unit volume) at each grid point of interest for use in KE
# calculation
KE_xplus = (uVar[:, :, p1]*uVar[:, :, p1] + vVar[:, :, p1]*vVar[:, :, p1] +
            wVar[:, :, p1]*wVar[:, :, p1])*0.5  # NG multiplied by 0.5 here
KE_xminus = (uVar[:, :, p2]*uVar[:, :, p2] + vVar[:, :, p2]*vVar[:, :, p2] +
             wVar[:, :, p2]*wVar[:, :, p2])*0.5  # NG multiplied by 0.5 here
KE_zminus = (uVar[:, p3, :]*uVar[:, p3, :] + vVar[:, p3, :]*vVar[:, p3, :] +
             wVar[:, p3, :]*wVar[:, p3, :])*0.5  # NG multiplied by 0.5 here

# Assumption is that there is no velocity out of top of CV
# %%

#
#
# Flux calculations
#
#

# Fluxes in x-direction
# phi_xplus = (pVar[:, :, p1] + KE_xplus[:, :] + PE_xplus[:, :]
#              )*uVar[:, :, p1]  # [kg/s^3]or[W/m^2]
# phi_xminus = -(pVar[:, :, p2] + KE_xminus[:, :] + PE_xminus[:, :]
#                )*uVar[:, :, p2]
# NG: removed flux of PE
phi_xplus = (pVar[:, :, p1] +
             KE_xplus[:, :])*uVar[:, :, p1]  # [kg/s^3]or[W/m^2]
phi_xminus = -(pVar[:, :, p2] + KE_xminus[:, :])*uVar[:, :, p2]
del KE_xminus  # NG makes it easier on memory
del KE_xplus  # NG makes it easier on memory

# Flux in z-direction
# phi_zminus = -(pVar[:, p3, :] + KE_zminus[:, :] + PE_zminus[:, :])*(
#    wVar[:, p3, :])
# NG: removed flux of PE
phi_zminus = -(pVar[:, p3, :] + KE_zminus[:, :])*wVar[:, p3, :]
# Zero flux occurs through top surface
del KE_zminus  # NG makes it easier on memory

# integrate and add the three paths of energy flux (taking CW path)

PHI1 = np.trapz(phi_xplus[:, p3:], z[p3:], axis=1)  # NG: loops are evil
PHI2 = np.trapz(phi_xminus[:, p3:], z[p3:], axis=1)
PHI3 = np.trapz(phi_zminus[:, p2:p1+1], x[p2:p1+1], axis=1)
PHI = -PHI1 + PHI2 - PHI3
print('Done Flux')


# %%

#
#
# Calculation for extracted energy from front
#
#

EF = 0.*pVar[:, :, :]
EF[:, :, :] = M2[:, :]*(vVar[:, :, :]*wVar[:, :, :]/f)  # +
#                         uVar[:, :, :]*b[:, :, :]/N2[:, :])
# NG just removed the contribution from PE

EFtot = integrate_vol(x[p2:p1+1], z[p3:], EF[:, p3:, p2: p1+1:])

print('Done Extracted')
# %% NG removed PE

#
#
# Change in KE and NOT PE with time
#
#

# Calculating dKE/dt at every grid point within CV grid
dtKE = (uVar[:, :, :]*dudt[:, :, :] + vVar[:, :, :]*dvdt[:, :, :] +
        wVar[:, :, :]*dwdt[:, :, :])

del dudt  # NG makes it easier on memory
del dvdt
del dwdt

# Performing volume integral to get dKE/dt total for each time step
dtKEa = np.zeros(pVar[:, 1, :].shape)
dtKEtot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        dtKEa[i, k] = np.trapz(dtKE[i, p3:, k], z[p3:])

del dtKE  # NG makes it easier on memory

for i in range(len(pVar[:, 1, 1])):
    dtKEtot[i] = np.trapz(dtKEa[i, p2:p1+1], x[p2:p1+1])

del dtKEa  # NG makes it easier on memory

# NG NOT Calculating dPE/dt at every grid point within CV grid
# dtPE = np.zeros(pVar[:, :, :].shape)
# dtPE[:, :, :] = (-g/(rho0*N2[:, :]))*b[:, :, :]*ds1dt[:, :, :]

# NG NOT Performing volume integral to get dPE/dt total for each time step
# dtPEa = np.zeros(pVar[:, 1, :].shape)
# dtPEtot = np.zeros(401)
# for i in range(len(pVar[:, 1, 1])):
#    for k in range(nx[0], nx[len(nx)-1]+1):
#        dtPEa[i, k] = np.trapz(dtPE[i, p3:, k], z[p3:])
# for i in range(len(pVar[:, 1, 1])):
#    dtPEtot[i] = np.trapz(dtPEa[i, p2:p1+1], x[p2:p1+1])

dEdt = dtKEtot  # +dtPEtot

print('Done dEdt')
# %%

#
#
# Horizontal derivatives
#
#
# NG correct begin
# uVar1 = uVar[:, :, :] +ThW[:, :]
vVar1 = vVar[:, :, :] + ThW[:, :]
# wVar1 = wVar[:, :, :] +ThW[:, :]
# bVar1 = -g*s1Var/rho0  # NG add
del s1Var

k = tf.k_of_x(x)
K, M = np.meshgrid(k, z)
print('Start Fourier')
FFF = tf.obfft(x, uVar, 2)  # NG correct
DFF = FFF*(1j*K)**2
d2udx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4udx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier 33%')
FFF = tf.obfft(x, vVar1, 2)
del vVar1
DFF = FFF*(1j*K)**2
d2vdx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4vdx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier  67%')
FFF = tf.obfft(x, wVar, 2)  # NG correct
DFF = FFF*(1j*K)**2
d2wdx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4wdx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier 100%')
# FFF = tf.obfft(x, bVar1, 2)  # NG correct
# DFF = FFF*(1j*K)**2
# d2s1dx2 = np.real(tf.obifft(k, DFF, 2))
# DFF = FFF*(1j*K)**4
# d4s1dx4 = np.real(tf.obifft(k, DFF, 2))
# print('Done Fourier 100%')
# end NG correct

del FFF
del DFF

# %%

#
#
# Energy dissipation calculations
#
#

# Background velocity profile (513, 1024)
d2ThWdz2 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*dltH*Lz)
# d4ThWdz4 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*(dltH**3)*(Lz**3))

# Add background profile to fluctutations
# d2udz2 = d2udz2[:, :, :] + d2ThWdz2[:, :]
# d4udz4 = d4udz4[:, :, :] + d4ThWdz4[:, :]
d2vdz2 = d2vdz2[:, :, :] + d2ThWdz2[:, :]
del d2ThWdz2
# NG: no hyperviscous operator on the thermal wind (because I wrote is so)
# d4vdz4 = d4vdz4[:, :, :] + d4ThWdz4[:, :]
# d2wdz2 = d2wdz2[:, :, :] + d2ThWdz2[:, :]
# d4wdz4 = d4wdz4[:, :, :] + d4ThWdz4[:, :]

udisp1 = uVar[:, :, :]*(nuh1*d2udx2[:, :, :] + nuz1*d2udz2[:, :, :] -
                        nuh2*d4udx4[:, :, :] - nuz2*d4udz4[:, :, :])
del uVar  # NG makes it easier on memory; that was the last uVar
del d2udx2
del d4udx4
del d2udz2
del d4udz4

vdisp1 = vVar[:, :, :]*(nuh1*d2vdx2[:, :, :] + nuz1*d2vdz2[:, :, :] -
                        nuh2*d4vdx4[:, :, :] - nuz2*d4vdz4[:, :, :])
del vVar  # NG makes it easier on memory; that was the last vVar
del d2vdx2
del d4vdx4
del d2vdz2
del d4vdz4

wdisp1 = wVar[:, :, :]*(nuh1*d2wdx2[:, :, :] + nuz1*d2wdz2[:, :, :] -
                        nuh2*d4wdx4[:, :, :] - nuz2*d4wdz4[:, :, :])
del wVar  # NG makes it easier on memory; that was the last vVar
del d2wdx2
del d4wdx4
del d2wdz2
del d4wdz4

# s1disp1 = (nuh1*d2s1dx2[:, :, :] + nuz1*d2s1dz2[:, :, :] -
#            nuh2*d4s1dx4[:, :, :] - nuz2*d4s1dz4[:, :, :])*(
#            -g*b[:, :, :]/rho0/N2[:, :])  # that's wrong I think

print('Start Disp')

# Performing surface integrals to get total dissipation at each time step
udisp1a = np.zeros(pVar[:, 1, :].shape)
udisp1tot = np.zeros(401)
# for i in range(len(pVar[:, 1, 1])):
#     for k in range(nx[0], nx[len(nx)-1]+1):
#         udisp1a[i, k] = np.trapz(udisp1[i, p3:, k], z[p3:])
# NG: loops are evil
udisp1a[i, k] = np.trapz(udisp1[i, p3:, k], z[p3:], axis=1)

del udisp1  # NG

for i in range(len(pVar[:, 1, 1])):
    udisp1tot[i] = np.trapz(udisp1a[i, p2:p1+1], x[p2:p1+1])

del udisp1a  # NG
print('Done Disp  33%')  # NG correct

vdisp1a = np.zeros(pVar[:, 1, :].shape)
vdisp1tot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        vdisp1a[i, k] = np.trapz(vdisp1[i, p3:, k], z[p3:])

del vdisp1  # NG

for i in range(len(pVar[:, 1, 1])):
    vdisp1tot[i] = np.trapz(vdisp1a[i, p2:p1+1], x[p2:p1+1])

del vdisp1a  # NG
print('Done Disp  67%')  # NG correct

wdisp1a = np.zeros(pVar[:, 1, :].shape)
wdisp1tot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        wdisp1a[i, k] = np.trapz(wdisp1[i, p3:, k], z[p3:])

del wdisp1  # NG

for i in range(len(pVar[:, 1, 1])):
    wdisp1tot[i] = np.trapz(wdisp1a[i, p2:p1+1], x[p2:p1+1])

del wdisp1a  # NG
print('Done Disp 100%')  # NG correct

# NG removed
# s1disp1a = np.zeros(pVar[:, 1, :].shape)
# s1disp1tot = np.zeros(401)
# for i in range(len(pVar[:, 1, 1])):
#     for k in range(nx[0], nx[len(nx)-1]+1):
#         s1disp1a[i, k] = np.trapz(s1disp1[i, p3:, k], z[p3:])
# for i in range(len(pVar[:, 1, 1])):
#     s1disp1tot[i] = np.trapz(s1disp1a[i, p2:p1+1], x[p2:p1+1])

del pVar

Dtot = udisp1tot + vdisp1tot + wdisp1tot  # +s1disp1tot

del udisp1tot  # NG
del vdisp1tot  # NG
del wdisp1tot  # NG

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
# Saving
if save == 1:
    np.save(setup+"PHI.npy", PHI)
    np.save(setup+"EFtot.npy", EFtot)
    np.save(setup+"dEdt.npy", dEdt)
    np.save(setup+"Dtot.npy", Dtot)
    np.save(setup+"Ctot.npy", Ctot)
    np.save(setup+"Ebudget.npy", Ebudget)

openXZ.close()
openderivs.close()
# opent_derivs.close()
