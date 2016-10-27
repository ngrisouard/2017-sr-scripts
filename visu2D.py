import pickle
import numpy as np
# import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import os
import sys
from pylab import get_current_fig_manager
# import scipy.io.netcdf as nc
import netCDF4 as nc
# import time
if '/Users/NicoG/Dropbox/python_useful' not in sys.path:
    sys.path.append('/Users/NicoG/Dropbox/python_useful')
if '/Users/NicoG/Dropbox/python_useful/obngfft' not in sys.path:
    sys.path.append('/Users/NicoG/Dropbox/python_useful/obngfft')
import FD
import ngobfft as TF

plt.close('all')


# %%
class dictionary_to_class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# %%
def read_field_2D(VrNm, flpath, numproc, rectype):
    if numproc == 1:
        fileID = nc.Dataset("{}.nc".format(flpath), 'r')
        if rectype == 'new':
            U = fileID.variables[VrNm][:, :].copy()
        else:
            U = fileID.variables[VrNm][:, :, :].copy()
    else:
        fileID = nc.MFDataset("{}*.nc".format(flpath))

    return U

base = '/Users/NicoG/FS_Morgan/outputs/'
# base = '/Volumes/LaCie2TB/M1/'
setup = 'N50F10R09-02/'
fl = 'XZ'

var = 's1Var'  # U, V, W, B, KE, PE, E, P or EPV
ARFac = 1  # aspect ratio factor for plotting (1: nx/nz sort of ratio)
isopyc = True   # 1 to plot isopycnals
quiv = False  # 1 to plot the velocity field as quiver plot
qxsk, qzsk = 1, 40   # one arrow every qxsk, qzsk in x, z
skx, skz = 1, 1
sk_us = 10
movie_y_n = True
print_o_n = False
four_o_n = False
ftsz = 12
lnwt = 2
ini_per = 0
accur = 2
lvl = 500
ncont = 64

# %% Load variables ------------------------------------------------------ # %%

root = '{0}{1}'.format(base, setup)

pkl_file = open('{0}spec_params.pkl'.format(root), 'rb')
d2pkl = pickle.load(pkl_file)
pkl_file.close()

s = dictionary_to_class(**d2pkl)
# local().update(d2pkl)
del d2pkl

# %% Compute additional quantities --------------------------------------- # %%

omega = 2*np.pi/s.T_omega

# --- Derivation operators
Dz1 = FD.getFDMatrix(s.nnz, s.dz, 1, accur)
Dz1 = Dz1.T
Dz2 = FD.getFDMatrix(s.nnz, s.dz, 2*s.hdeg, accur)
Dz2 = Dz2.transpose()
Dz3 = FD.getFDMatrix(s.nnz, s.dz, 2*s.hdeg+1, accur)
Dz3 = Dz3.transpose()

Dx1 = FD.getFDMatrix(s.nx, s.dx, 1, accur)
Dx2 = FD.getFDMatrix(s.nx, s.dx, 2*s.hdeg, accur)
Dx3 = FD.getFDMatrix(s.nx, s.dx, 2*s.hdeg+1, accur)

# A = np.exp(s.zxs/250.)
# B = Dz1.T.dot(A)

# --- This one
kmat = np.tile(s.k, [1, s.nnz])

# --- Time loop
itearr = range(0, s.nites, sk_us*s.sk_it)

if not os.path.exists('{0}figures'.format(root)) and print_o_n:
    os.mkdir('{0}figures'.format(root))

# %% Open netcdf file(s) ------------------------------------------------- # %%

if s.RecType != 'append':
    raise NameError('Record type is not append')

# fID = nc.MFDataset('{0}2D/XZ{1}*.nc'.format(root, s.iteID, s.npID), 'r')
if s.np == 1:
    fID = nc.Dataset('{0}2D/XZ.nc'.format(root), 'r')
else:
    raise NameError("More than one processor")

# %% Compute various stuff ----------------------------------------------- # %%

AspRat = ARFac*(s.Lx/s.nx)/(s.Lz/s.nz)  # aspect ratio for plotting

plt.ioff()
vrbl = fID.variables[var][::sk_us, :, :].copy()
if four_o_n:
    TFvrbl = TF.obfft(s.x, vrbl, 2)
    TFvrbl /= np.nanmax(abs(TFvrbl))

for ii in range(vrbl.shape[0]):
    print('ii = ' + str(ii))
    # vrbl = fID.variables[var][ii, ::skz, ::skx].copy()

    plt.close(1)
    fig1 = plt.figure(1)
    ax1 = fig1.gca()
    pcax1 = ax1.pcolormesh(s.xz, s.zx, vrbl[ii, :, :])
    plt.colorbar(pcax1)
    ax1.axis([s.x.min(), s.x.max(), 0, s.Lz])
    ax1.set_aspect(AspRat)
    plt.tight_layout()
    plt.show(1)

    plt.pause(5)

    get_current_fig_manager().window.raise_()

    if four_o_n:
        # TFvrbl = TF.obfft(s.x[::skx], vrbl, 1)

        plt.close(2)
        fig2 = plt.figure(2)
        ax2 = fig2.gca()
        pcax2 = ax2.pcolormesh(s.km, s.zx[:-1, :],
                               np.log10(abs(TFvrbl[ii, :-1, :])),
                                            vmin = -16., vmax = 0.)
        plt.colorbar(pcax2)
        plt.show(2)

        get_current_fig_manager().window.raise_()
        plt.pause(0.1)

fID.close()
