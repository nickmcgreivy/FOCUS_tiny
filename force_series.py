from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import h5py
import tables as tb
import numpy as numpy
from objective_function import objective_and_gradient,  make_array, array2I_arr, array2both, array2p
from objective_function import quadratic_flux_error, total_energy, toroidal_flux, coil_length
from objective_function import config as cfg

from surface import Surface
from jax import jit, grad, vmap
from functools import partial
import scipy.optimize as optimize
from jax.ops import index_update, index
from mayavi import mlab


def draw_surface(r_surf):
    r_surf = np.concatenate((r_surf, r_surf[0:, :, :]), axis=0)
    r_surf = np.concatenate((r_surf, r_surf[:, 0:, :]), axis=1)
    x = r_surf[:,:,0]
    y = r_surf[:,:,1]
    z = r_surf[:,:,2]
    p = mlab.mesh(x,y,z,color=(0.8,0.0,0.0))
    return p


def draw_coils(ll):
    for ic in range(ll.shape[0]):
        p = mlab.plot3d(ll[ic,:,0], ll[ic,:,1], ll[ic,:,2], tube_radius = 0.004, line_width = 0.05, color = (0.0, 0.0, 0.8))
    return p


def r(p, theta):
    """
    get cartesian positions from a coilSeries object through fourier
    unpacking.
    See Focus documentation for how the fourier components are packed.
    Arguments:
    *p*: (6,n_coils,m)-array containing the sine and cosine series describing the coils. first index
    determines (.
    xcosine, ycosine, zcosine, xsine, ysine, zsine)
    *theta*: series of angles where the fourier series is to be evaluated.
    Returns:
    *r*: (n_coils, n_segments+1,3)-array of positions on each coil, where last element equals the first
    """
    r = np.zeros((3, p.shape[1], cfg.NS + 1))
    for m in range(p.shape[2]):
        r += p[:3, :, None, m] * np.cos(m * theta)[None, None, :] + \
            p[3:, :, None, m] * np.sin(m * theta)[None, None, :]
    return np.transpose(r, (1, 2, 0))

#######################################################################
# Reading in coil and surface data
#######################################################################
N=0
cfg.NS = 32
cfg.N = 100
cfg.lr = 0.001
cfg.theta = np.linspace(0, 2 * np.pi, cfg.NS + 1)
cfg.nn = np.load("nn.npy")
cfg.sg = np.load("sg.npy")
cfg.r_surf = np.load("r_surf.npy")
with h5py.File("coils.hdf5", "r") as f:
    p = np.asarray(f['coilSeries'])
    n_coils = f["metadata"]["NC"]
# optional: zero-pad the p-array for higher fourier resolution
cfg.p_shape = p.shape
cfg.p_size = p.size
cfg.pad_length = cfg.NS - p.shape[-1]  # length to pad the fourier transform with

I_arr = np.ones(n_coils)
cfg.I_arr_shape = I_arr.shape
cfg.I_arr_size = I_arr.size


i=1
for energy_weight in np.logspace(-7, -2, 10):
    mlab.clf()
    print("making figure for weight {}".format(energy_weight))
    filename = "coils_force_{}.hdf5".format(energy_weight)
    with h5py.File(filename, "r") as f:
        p = np.asarray(f['coilSeries'])
        I_arr = np.asarray(f['I_arr'])
    mlab.clf()
    ll = r(p, cfg.theta)
    q = draw_coils(ll)
    q = draw_surface(cfg.r_surf)
    #mlab.show()
    mlab.savefig('force_series_{}.png'.format(i), size = (2000,2000))
    i+=1



