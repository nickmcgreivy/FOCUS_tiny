import jax.numpy as np
from jax import jit, grad, vmap, pmap
from jax.config import config
import tables as tb
import time
from functools import partial

NS = 32
N = 10
lr = 0.0001
theta = np.linspace(0, 2 * np.pi, NS + 1)
nn = np.load("nn.npy")
sg = np.load("sg.npy")
r_surf = np.load("r_surf.npy")
with tb.open_file("coils.hdf5", "r") as f:
	fc = np.asarray(f.root.coilSeries[:, :, :])

#######################################################################

def biot_savart(r_eval, dl, l):
	top = np.cross(dl, r_eval[None, None, :] - l)
	bottom = np.linalg.norm(r_eval[None, None, :] - l, axis=-1) ** 3
	B = np.sum(top / bottom[:, :, None], axis=(0, 1))
	return B

def quadratic_flux(r, dl, l, nn, sg):
	alpha = biot_savart_surface(r, dl, l)
	return (0.5 * np.sum(np.sum(nn * alpha, axis=-1) ** 2 * sg))

def r(fc, theta):
	r = np.zeros((3, fc.shape[1], NS + 1))
	for m in range(fc.shape[2]):
		r += fc[:3, :, None, m] * np.cos(m * theta)[None, None, :] + fc[3:, :, None, m] * np.sin(m * theta)[None, None, :]
	return np.transpose(r, (1, 2, 0))

def loss(r_surf, nn, sg, weight, fc):
	l = r(fc, theta)
	return quadratic_flux(r_surf, l[:,:-1,:] - l[:,1:,:], l[:,:-1,:], nn, sg) + weight * np.sum(dl)

#######################################################################

 # functional programming (Python)
objective_function = partial(loss, r_surf, nn, sg, 0.1)

# vectorize (JAX)
biot_savart_surface = vmap(vmap(biot_savart, (0, None, None), 0), (1, None, None), 1)

# automatic differentiation (JAX)
grad_func = grad(objective_function) # d output / d input

# jit-compile (JAX)
jit_grad_func = jit(grad_func)

#######################################################################

print("loss is {}".format(objective_function(fc)))

for n in range(N):
	grad = jit_grad_func(fc)
	fc = fc - grad * lr
	print(n)
		
print("loss is {}".format(objective_function(fc)))
