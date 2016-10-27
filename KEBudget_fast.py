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
    print('{0} completed, elapsed time = {1: 4d} s'.format(str_step,
          int(time.time() - tic)))
    return time.time()


def print_int_elpsd(str_step, counter, ncounters, tic):
    print('    {0}: {1: >3d}% done, {2: 4d} s elapsed'.format(str_step,
          int((100*counter)/ncounters), int(time.time() - tic)))
    counter += 1
    return counter

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
p1 = 850  # right boundar
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
Var['u'] = openXZ.variables['uVar'][:, :, :].copy()
Var['v'] = openXZ.variables['vVar'][:].copy()
Var['w'] = openXZ.variables['wVar'][:].copy()
Var['s1'] = openXZ.variables['s1Var'][:].copy()

tic = print_elpsd('Loading Vars', tic)


# %% NG: I'm not sure if this is in the right place
# Saving
# if not save:
#    PHI = np.load("PHI.npy")
#    EFtot = np.load("EFtot.npy")
#    dEdt = np.load("dEdt.npy")
#    Dtot = np.load("Dtot.npy")
#    Dtot = np.load("Ctot.npy")
#    Ebudget = np.load("Ebudget.npy")

# %%
t = np.zeros(401)
t = np.linspace(0, 400*dt, 401)  # NG: faster this way

# x = np.linspace(0, Lx, nx_FS)
# z = np.linspace(-Lz, 0, nz_FS+1)
x = openXZ.variables['xVar'][:].copy()  # The linspace thing above isn't
z = openXZ.variables['zVar'][:].copy()  # exactly the actual x...

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

# %%

#
#
# PE calculation
#
#

# calculate buoyancy forces
# Var['b'] = -g*(Var['s1'][:, :, :] - Var['s1'][0, :, :][:, :])/rho0
Var['b'] = -g*(Var['s1'][:, :, :] - s1ref[:, :])/rho0

# NG added conversion KE to PE

conv = - Var['b']*Var['w']

Ctot = integrate_vol(x[p2:p1+1], z[p3:], conv[:, p3:, p2:p1+1])
del conv

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

pVar_xplus = openderivs.variables['pVar'][:, :, p1].copy()
phi_xplus = (pVar_xplus +
             0*KE_xplus[:, :])*Var['u'][:, :, p1]  # [kg/s^3]or[W/m^2]

pVar_xminus = openderivs.variables['pVar'][:, :, p2].copy()
phi_xminus = -(pVar_xminus + 0*KE_xminus[:, :])*Var['u'][:, :, p2]

pVar_zminus = openderivs.variables['pVar'][:, p3, :].copy()
phi_zminus = -(pVar_zminus + 0*KE_zminus[:, :])*Var['w'][:, p3, :]

# sys.exit()

# Zero flux occurs through top surface

# integrate and add the three paths of energy flux (taking CW path)
PHI1 = np.trapz(phi_xplus[:, p3:], z[p3:], axis=1)  # NG: loops are evil
PHI2 = np.trapz(phi_xminus[:, p3:], z[p3:], axis=1)
PHI3 = np.trapz(phi_zminus[:, p2:p1+1], x[p2:p1+1], axis=1)
PHI = + PHI1 + PHI2 + PHI3

tic = print_elpsd('Fluxes', tic)


# %%

#
#
# Calculation for Geostrophic Shear Production (GSP)
#
#

GSP = 0.*Var['u']
GSP[:, :, :] = M2[:, :]*Var['v'][:, :, :]*Var['w'][:, :, :]/f  # +
#                         Var['u'][:, :, :]*b[:, :, :]/N2[:, :])
# NG just removed the contribution from PE

GSPtot = integrate_vol(x[p2:p1+1], z[p3:], GSP[:, p3:, p2:p1+1])
del GSP

tic = print_elpsd('GSP', tic)


# %%

#
#
# Calculation forLateral Shear Production (LSP)
#
#

LSP = 0.*Var['u']
LSP[:, :, :] = dThWdx[:, :]*Var['v'][:, :, :]*Var['u'][:, :, :]

LSPtot = integrate_vol(x[p2:p1+1], z[p3:], LSP[:, p3:, p2:p1+1])
del LSP

tic = print_elpsd('LSP', tic)

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
print('Start Dissipation')

for ii in ['u', 'v', 'w']:
    """ First compute and add each term. Once done, integrate the result.
    Looks like more loops than necessary, but I find it easier to read. """
    for jj in ['2', '4']:
        dzNm = 'd' + jj + ii + 'dz' + jj
        dissip += dffs['nuz'+jj] * add_deriv_load(openderivs, Var[ii], dzNm)
        dissip += dffs['nuh'+jj] * add_deriv_four(x, Var[ii], K, int(jj))
        counter = print_int_elpsd('Dissipation', counter, 6, tic)

# dissipation of the thermal wind
dissip = dissip[:, :, :] + (nuz*d2ThWdz2[:, :] + nuh*d2ThWdx2[:, :]
                            )*Var['v'][:, :, :]

Dtot = integrate_vol(x[p2:p1+1], z[p3:], dissip[:, p3:, p2:p1+1])
del dissip

tic = print_elpsd('Dissipation', tic)


# %%

#
#
# Full accounting of Energy Budget -> checking for LHS=RHS and all energy is
# conserved
#
#

Ebudget = -PHI - GSPtot - LSPtot - dEdt + Dtot - Ctot

fig1 = plt.figure()
plt.plot(t, -PHI1, 'k', label='x+ Flux')
plt.plot(t, -PHI2, 'k--', label='x- Flux')
plt.plot(t, -PHI3, 'k+-', label='z- Flux')
plt.plot(t, -GSPtot, 'm', label='GSP')
plt.plot(t, -LSPtot, 'm--', label='LSP')
plt.plot(t, -dEdt, 'b', label='dEdt')
plt.plot(t, Dtot, 'g', label='Dissipation')
plt.plot(t, -Ctot, 'c', label='-Conversion')
plt.plot(t, Ebudget, 'r--', label='Cumulative Budget', lw=2)

plt.xlabel("Time [$s$]")
plt.ylabel("Integrated Energy Per Unit Density [$m^4/s^3$]")
plt.title("Full Accounting of Energy Budget")
plt.grid()
plt.legend(loc='upper left')
plt.show()


# %%
# Saving and concluding

Var = {}

if save == 1:
    np.save(setup+"PHI.npy", PHI)
    np.save(setup+"GSPtot.npy", GSPtot)
    np.save(setup+"LSPtot.npy", LSPtot)
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
print('All done! Elapsed time = {0:3d}:{0:02d}'.format(mins, secs))
print(' ')
print('          ******')
print(' ')
