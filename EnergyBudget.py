import sys
import matplotlib.pyplot as plt
import numpy as np
# from numpy import shape
from scipy.io import netcdf
import time
import getpass

if getpass.getuser() == 'NicoG':
    homedir = '/Users/NicoG'
    obftdir = '/Dropbox/python_useful'
    base = homedir + '/FS_in-house/outputs/'
elif getpass.getuser() == 'nicolas':
    homedir = '/home/nicolas'
    obftdir = homedir + '/Dropbox/python_useful'
    base = '/data/nicolas/mbf/'
elif getpass.getuser() == 'mbfox':
    homedir = '/home/mbfox'
    obftdir = homedir + '/Desktop'
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the home directory?')

if '{}/Dropbox/python_useful'.format(homedir) not in sys.path:
    sys.path.append('{}/Dropbox/python_useful'.format(homedir))

if '{}/obngfft'.format(obftdir) not in sys.path:
    sys.path.append('{}/obngfft'.format(obftdir))

if (sys.version_info > (3, 0)):  # Then Python 3 is running
    import ngobfftPy3 as tf
else:  # Then is has to be Python 2
    import ngobfft as tf

# %% NG start
nx_FS = 1024
nz_FS = 512
p1 = 850  # don't know what this is yet
p2 = 713  # idem
p3 = 280  # ibidem

# Below I repatriated stuff that were defined throughout the text

save = 1  # 1 to compute and save, 0 to just load

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

XZVarList = list(openXZ.variables.keys())

# With Python3, one has to convert the dict_keys into a list
print(openXZ.variables[XZVarList[1]])

# Converting netcdf to dictionary with variable names as the key
strpData = {}
for i in range(0, len(XZVarList)):
    temp = openXZ.variables[XZVarList[i]]
    varData = temp[:].copy()
    strpData[XZVarList[i]] = varData

uVar = strpData['uVar']    # X velocity (401, 513, 1024)
vVar = strpData['vVar']    # V velocity
wVar = strpData['wVar']    # Z velocity
s1Var = strpData['s1Var']  # Density data

openXZ.close()

print('Loading XZ completed, elapsed time = {}'.format(time.time() - tic))
tic = time.time()

# %% Opening the Netcdf file
openderivs = netcdf.netcdf_file(derivsfile, 'r')

# With Python3, one has to convert the dict_keys into a list
derivsVarList = list(openderivs.variables.keys())

print(openderivs.variables[derivsVarList[1]])

# Converting netcdf to dictionary with variable names as the key
strpData = {}
for i in range(0, len(derivsVarList)):
    temp = openderivs.variables[derivsVarList[i]]
    varData = temp[:].copy()
    strpData[derivsVarList[i]] = varData

pVar = strpData['pVar']  # Pressure data (401, 513, 1024)
d2udz2 = strpData['d2udz2']
d4udz4 = strpData['d4udz4']
d2vdz2 = strpData['d2vdz2']
d4vdz4 = strpData['d4vdz4']
d2wdz2 = strpData['d2wdz2']
d4wdz4 = strpData['d4wdz4']
d2s1dz2 = strpData['d2s1dz2']
d4s1dz4 = strpData['d4s1dz4']

openderivs.close()

print('Loading derivs completed, elapsed time = {}'.format(time.time() - tic))
tic = time.time()

# %% Opening the Netcdf file
opent_derivs = netcdf.netcdf_file(t_derivsfile, 'r')

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
ds1dt = strpData['ds1dt']  # s1 time derivative data

openderivs.close()

print('Loading tderivs completed, elapsed time = {}'.format(time.time() - tic))
tic = time.time()


# %%
# Saving
if not save:
    PHI = np.load("PHI.npy")
    EFtot = np.load("EFtot.npy")
    dEdt = np.load("dEdt.npy")
    Dtot = np.load("Dtot.npy")
    Ebudget = np.load("Ebudget.npy")

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

# N^2 is the vertical buyoancy frequency
N2_1 = N2_0 / (1 - Z / z0Gill) ** 2
N2 = N2_1 + Bhat * Gamma * np.exp(Z / dltH / Lz) / dltH / Lz

# Background velocity profile (513,1024)
ThW = Bhat*dGammadx*dltH*Lz*((np.exp(Zshft/(dltH*Lz)) -
                             np.exp(-totdepth/(dltH*Lz))))/f

# %%
#
#
# KE Calculation
#
#

# Calculating KE(per unit volume) at each grid point of interest for use in KE
# calculation
KE_xplus = (uVar[:, :, p1]*uVar[:, :, p1] + vVar[:, :, p1]*vVar[:, :, p1] +
            wVar[:, :, p1]*wVar[:, :, p1])*0.5
KE_xminus = (uVar[:, :, p2]*uVar[:, :, p2] + vVar[:, :, p2]*vVar[:, :, p2] +
             wVar[:, :, p2]*wVar[:, :, p2])*0.5
KE_zminus = (uVar[:, p3, :]*uVar[:, p3, :] + vVar[:, p3, :]*vVar[:, p3, :] +
             wVar[:, p3, :]*wVar[:, p3, :])*0.5

# Assumption is that there is no velocity out of top of CV

# %%

#
#
# PE calculation
#
#

# calculate buoyancy forces
b = ((s1Var[:, :, :])/rho0)*(-g)

PE_xplus = (b[:, :, p1]*b[:, :, p1])/N2[:, p1]*0.5
PE_xminus = (b[:, :, p2]*b[:, :, p2])/N2[:, p2]*0.5
PE_zminus = (b[:, p3, :]*b[:, p3, :])/N2[p3, :]*0.5

# %%

#
#
# Flux calculations
#
#

# Fluxes in x-direction
phi_xplus = (pVar[:, :, p1] + KE_xplus[:, :] + PE_xplus[:, :]
             )*uVar[:, :, p1]  # [kg/s^3]or[W/m^2]
phi_xminus = -(pVar[:, :, p2] + KE_xminus[:, :] + PE_xminus[:, :]
               )*uVar[:, :, p2]

# Flux in z-direction
phi_zminus = -(pVar[:, p3, :] + KE_zminus[:, :] + PE_zminus[:, :])*(
    wVar[:, p3, :])
# Zero flux occurs through top surface

# integrate and add the three paths of energy flux (taking CW path)
PHI = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    PHI1 = np.trapz(phi_xplus[i, p3:], z[p3:])
    PHI2 = np.trapz(phi_xminus[i, p3:], z[p3:])
    PHI3 = np.trapz(phi_zminus[i, p2:p1+1], x[p2:p1+1])
    PHI[i] = -PHI1 + PHI2 - PHI3
print('Done Flux')

# %%

#
#
# Calculation for extracted energy from front
#
#

EF = np.zeros(pVar[:, :, :].shape)
EF[:, :, :] = M2[:, :]*(vVar[:, :, :]*wVar[:, :, :]/f +
                        uVar[:, :, :]*b[:, :, :]/N2[:, :])

EFtota = np.zeros(pVar[:, 1, :].shape)
EFtot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        EFtota[i, k] = np.trapz(EF[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    EFtot[i] = np.trapz(EFtota[i, p2:p1+1], x[p2:p1+1])

print('Done Extracted')
# %%

#
#
# Change in KE and PE with time
#
#

# Calculating dKE/dt at every grid point within CV grid
dtKE = (uVar[:, :, :]*dudt[:, :, :] + vVar[:, :, :]*dvdt[:, :, :] +
        wVar[:, :, :]*dwdt[:, :, :])

# Performing volume integral to get dKE/dt total for each time step
dtKEa = np.zeros(pVar[:, 1, :].shape)
dtKEtot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        dtKEa[i, k] = np.trapz(dtKE[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    dtKEtot[i] = np.trapz(dtKEa[i, p2:p1+1], x[p2:p1+1])

# Calculating dPE/dt at every grid point within CV grid
dtPE = np.zeros(pVar[:, :, :].shape)
dtPE[:, :, :] = (-g/(rho0*N2[:, :]))*b[:, :, :]*ds1dt[:, :, :]

# Performing volume integral to get dPE/dt total for each time step
dtPEa = np.zeros(pVar[:, 1, :].shape)
dtPEtot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        dtPEa[i, k] = np.trapz(dtPE[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    dtPEtot[i] = np.trapz(dtPEa[i, p2:p1+1], x[p2:p1+1])

dEdt = dtKEtot+dtPEtot

print('Done dEdt')
# %%

#
#
# Horizontal derivatives
#
#

uVar1 = uVar[:, :, :]+ThW[:, :]
vVar1 = vVar[:, :, :]+ThW[:, :]
wVar1 = wVar[:, :, :]+ThW[:, :]

k = tf.k_of_x(x)
K, M = np.meshgrid(k, z)
print('Start Fourier')
FFF = tf.obfft(x, uVar1, 2)
DFF = FFF*(1j*K)**2
d2udx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4udx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier 25%')
FFF = tf.obfft(x, vVar1, 2)
DFF = FFF*(1j*K)**2
d2vdx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4vdx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier 50%')
FFF = tf.obfft(x, wVar1, 2)
DFF = FFF*(1j*K)**2
d2wdx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4wdx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier 75%')
FFF = tf.obfft(x, s1Var, 2)
DFF = FFF*(1j*K)**2
d2s1dx2 = np.real(tf.obifft(k, DFF, 2))
DFF = FFF*(1j*K)**4
d4s1dx4 = np.real(tf.obifft(k, DFF, 2))
print('Done Fourier 100%')


# %%

#
#
# Energy dissipation calculations
#
#

nuh1 = 1e-5
nuh2 = 2e3
nuz1 = 1e-5
nuz2 = 1e-7

# Background velocity profile (513, 1024)
d2ThWdz2 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*dltH*Lz)
d4ThWdz4 = Bhat*dGammadx*np.exp(Zshft/(dltH*Lz))/(f*(dltH**3)*(Lz**3))

# Add background profile to fluctutations
d2udz2 = d2udz2[:, :, :] + d2ThWdz2[:, :]
d4udz4 = d4udz4[:, :, :] + d4ThWdz4[:, :]
d2vdz2 = d2vdz2[:, :, :] + d2ThWdz2[:, :]
d4vdz4 = d4vdz4[:, :, :] + d4ThWdz4[:, :]
d2wdz2 = d2wdz2[:, :, :] + d2ThWdz2[:, :]
d4wdz4 = d4wdz4[:, :, :] + d4ThWdz4[:, :]

udisp1 = uVar[:, :, :]*(nuh1*d2udx2[:, :, :] + nuz1*d2udz2[:, :, :] -
                        nuh2*d4udx4[:, :, :] - nuz2*d4udz4[:, :, :])
vdisp1 = vVar[:, :, :]*(nuh1*d2vdx2[:, :, :] + nuz1*d2vdz2[:, :, :] -
                        nuh2*d4vdx4[:, :, :] - nuz2*d4vdz4[:, :, :])
wdisp1 = wVar[:, :, :]*(nuh1*d2wdx2[:, :, :] + nuz1*d2wdz2[:, :, :] -
                        nuh2*d4wdx4[:, :, :] - nuz2*d4wdz4[:, :, :])
s1disp1 = (nuh1*d2s1dx2[:, :, :] + nuz1*d2s1dz2[:, :, :] -
           nuh2*d4s1dx4[:, :, :] - nuz2*d4s1dz4[:, :, :])*(
           -g*b[:, :, :]/rho0/N2_0)

print('Start Disp')

# Performing surface integrals to get total dissipation at each time step
udisp1a = np.zeros(pVar[:, 1, :].shape)
udisp1tot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        udisp1a[i, k] = np.trapz(udisp1[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    udisp1tot[i] = np.trapz(udisp1a[i, p2:p1+1], x[p2:p1+1])
print('Done Disp 25%')

vdisp1a = np.zeros(pVar[:, 1, :].shape)
vdisp1tot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        vdisp1a[i, k] = np.trapz(vdisp1[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    vdisp1tot[i] = np.trapz(vdisp1a[i, p2:p1+1], x[p2:p1+1])
print('Done Disp 50%')

wdisp1a = np.zeros(pVar[:, 1, :].shape)
wdisp1tot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        wdisp1a[i, k] = np.trapz(wdisp1[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    wdisp1tot[i] = np.trapz(wdisp1a[i, p2:p1+1], x[p2:p1+1])
print('Done Disp 75%')

s1disp1a = np.zeros(pVar[:, 1, :].shape)
s1disp1tot = np.zeros(401)
for i in range(len(pVar[:, 1, 1])):
    for k in range(nx[0], nx[len(nx)-1]+1):
        s1disp1a[i, k] = np.trapz(s1disp1[i, p3:, k], z[p3:])

for i in range(len(pVar[:, 1, 1])):
    s1disp1tot[i] = np.trapz(s1disp1a[i, p2:p1+1], x[p2:p1+1])

Dtot = udisp1tot+vdisp1tot+wdisp1tot+s1disp1tot

print('Done Disp 100%')
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

Ebudget = -PHI-EFtot-dEdt+Dtot
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
    np.save(setup+"Ebudget.npy", Ebudget)
