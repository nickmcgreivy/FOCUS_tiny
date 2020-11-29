from jax import numpy as np
import h5py

NS = 32
N = 100
lr = 0.001
theta = np.linspace(0, 2 * np.pi, NS + 1)
nn = np.load("nn.npy")
sg = np.load("sg.npy")
r_surf = np.load("r_surf.npy")
with h5py.File("coils.hdf5", "r") as f:
    p = np.asarray(f['coilSeries'])
    n_coils = f["metadata"]["NC"]

p_shape = p.shape
p_size = p.size
pad_length = NS - p.shape[-1]  # length to pad the fourier transform with

I_arr = np.ones(n_coils)
I_arr_shape = I_arr.shape
I_arr_size = I_arr.size
