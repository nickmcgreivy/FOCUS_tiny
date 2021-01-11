from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import h5py
#import tables as tb
import numpy as numpy
from objective_function import objective_and_gradient,  make_array, array2I_arr, array2both, array2p
from objective_function import quadratic_flux_error, total_energy, toroidal_flux, coil_length
from objective_function import config as cfg

from surface import Surface
from jax import jit, grad, vmap
from functools import partial
import scipy.optimize as optimize
from jax.ops import index_update, index

#######################################################################
# Reading in coil and surface data
#######################################################################

cfg.NS = 32
cfg.N = 100
cfg.lr = 0.001
cfg.theta = np.linspace(0, 2 * np.pi, cfg.NS + 1)
cfg.nn = np.load("nn.npy")
cfg.sg = np.load("sg.npy")
cfg.r_surf = np.load("r_surf.npy")
with h5py.File("force_success.hdf5", "r") as f:
    p = np.asarray(f['coilSeries'])
    #n_coils = f["metadata"]["NC"]
    n_coils = 20
# optional: zero-pad the p-array for higher fourier resolution
cfg.p_shape = p.shape
cfg.p_size = p.size
cfg.pad_length = cfg.NS - p.shape[-1]  # length to pad the fourier transform with

I_arr = np.ones(n_coils) * 0.8
cfg.I_arr_shape = I_arr.shape
cfg.I_arr_size = I_arr.size

#######################################################################
# Calculating objective function
######################################################################
#

objective, jit_grad_func = objective_and_gradient(1.0, 1e-5, 0.1, 1.0)

def callbackF(objective_array):
    Q = quadratic_flux_error(objective_array)
    E = total_energy(objective_array)
    L = coil_length(objective_array)
    T = toroidal_flux(objective_array, 1.0)
    print("Objective is now {}".format(objective(objective_array)))
    print("Q: {0: 3.6f}, E: {1: 3.6f}, L: {2: 3.6f}, T: {3: 3.6f}".format(Q, E, L, T))

objective_arrays = []
objective_array = make_array(p, I_arr)

#print("Starting loss is {}".format(objective(objective_array)))
#res = optimize.minimize(objective, objective_array, jac=jit_grad_func, callback=None)
#objective_array = res.x
#callbackF(res.x)
results = []


for energy_weight in [.0003,]:
    print("energy weight is now {}".format(energy_weight))
    objective, jit_grad_func = objective_and_gradient(1.0, energy_weight, 0.1, 1.0)
    res = optimize.minimize(objective, objective_array, jac=jit_grad_func, callback=None)
    # objective_array = res.x
    callbackF(res.x)
    results.append(res)
#    with tb.open_file("coils_force_{}.hdf5".format(energy_weight), "w") as f:
#        f.create_array("/", "coilSeries", numpy.asarray(array2p(res.x)))
#        f.create_array("/", "I_arr", numpy.asarray(array2I_arr(res.x)))





print('done')
