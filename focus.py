import jax.numpy as np
import h5py
from jax import jit, grad, vmap
from jax.config import config
import tables as tb
from functools import partial
import numpy as numpy
import scipy.optimize as optimize

#######################################################################
# Reading in coil and surface data
#######################################################################

NS = 32
N = 300
lr = 0.001
theta = np.linspace(0, 2 * np.pi, NS + 1)
nn = np.load("nn.npy")
sg = np.load("sg.npy")
r_surf = np.load("r_surf.npy")
with h5py.File("coils.hdf5", "r") as f:
    p = np.asarray(f['coilSeries'])
    n_coils = f["metadata"]["NC"]
I_arr = np.ones(n_coils)

#######################################################################
# Calculating objective function
#######################################################################


def biot_savart(r_eval, dl, ll, I_arr):
    """
    Calculate the complete Biot-Savart integral over the coils
    specified by l and dl.
    Arguments:
    *r_eval*: (lenght 3 array) the point wherer the field is to be evaluated in cartesian
    coordinates.
    *dl*: ( n_coils, nsegments, 3)-array of the distance vector to every
    other coil line segment
    *l* ( n_coils, nsegments, 3)-array of the position of each coil segment
    *I_arr*: (n_coils)-array specifying the current per coil.

    Note on algoritnm: the None allows one to add new axes to in-line
    cast the array into the proper shape.
    The biot-savart integral is calculated as a sum over all segments.

    returns:
    *B*: magnetic field at position r_eval
    """
    top = np.cross(dl, r_eval[None, None, :] - ll) * I_arr[:, None,
                                                           None]  #unchecked
    bottom = np.linalg.norm(r_eval[None, None, :] - ll, axis=-1)**3
    B = np.sum(top / bottom[:, :, None], axis=(0, 1))
    return B


def biot_savart_oncoil(r_eval, dl, ll, I_arr):
    """
    Calculate the Biot-Savart integral over the coils (also ON) a segment of the
    coil.
    specified by l and dl.
    Arguments:
    *r_eval*: (lenght 3 array) the point wherer the field is to be evaluated in cartesian
    coordinates. Has to be on a coil.
    *dl*: ( n_coils, nsegments, 3)-array of the distance vector to every
    other coil line segment
    *l* ( n_coils, nsegments, 3)-array of the position of each coil segment

    Note on algoritnm: the None allows one to add new axes to in-line
    cast the array into the proper shape.
    The biot-savart integral is calculated as a sum over all segments.

    returns:
    *B*: magnetic field at position r_eval
    """
    top = np.cross(dl, r_eval[None, None, :] - ll) * I_arr[:, None,
                                                           None]  #unchecked
    bottom = np.linalg.norm(r_eval[None, None, :] - ll, axis=-1)**3
    # sum over all infinitesimal line segments, replacing the NaN with zero
    B = np.sum(np.nan_to_num(top / bottom[:, :, None]), axis=(0, 1))
    return B


def vector_potential(r_eval, ll, dl, I_arr):
    """
    calculate the vector potential in the Lorentz Gauge at position r_eval
    Arguments:
    *r_eval*: (lenght 3 array) the point wherer the field is to be evaluated in cartesian
    coordinates.
    *dl*: ( n_coils, nsegments, 3)-array of the distance vector to every
    other coil line segment
    *l* ( n_coils, nsegments, 3)-array of the position of each coil segment
    returns:
    *A*: Vector potential at position r_eval

    equation: $\mathbf{A} = \sum_coils \oint I\mathrm{d}
    """
    top = dl * I_arr[:, None, None]
    bottom = np.linalg.norm(r_eval[None, None, :] - ll, axis=-1)
    A = np.sum(top / bottom[:, :, None], axis=(0, 1))
    return A


def flux_through_all_loops(r_surf, ll, dl, I_arr):
    """
    returns the magnetic flux through all poloidal and toroidal loops
    of the surface array.
    Arguments:
    *r_surf*: (n, m, 3) array of cartesian coordinates on the surface.
    [:,0,:] must be the first poloidal loop, and [0,:,:] the first toroidal loop.
    *dl*: ( n_coils, nsegments, 3)-array along the coil segments.
    other coil line segment
    *l* ( n_coils, nsegments, 3)-array of the position of each coil segment
    returns:
    *polint*: n-array of poloidal fluxes
    *torint*: m-array of toroidal fluxes
    """
    mapped_vector_potential = vmap(
        vmap(vector_potential, (0, None, None, None), 0),
        (1, None, None, None), 1)
    vec_surf = mapped_vector_potential(r_surf, ll, dl, I_arr)
    dl_pol = np.hstack((r_surf[:, 1:, :] - r_surf[:, :-1, :],
                        (r_surf[:, 0, :] - r_surf[:, -1, :])[:, None, :]))
    A_midpol = 0.5 * np.hstack(
        (vec_surf[:, 1:, :] + vec_surf[:, :-1, :],
         (vec_surf[:, 0, :] + vec_surf[:, -1, :])[:, None, :]))
    # sum over axis 0 and 2 only, the product of the two. Calculates the inner product, and then "integrates" poloidally.
    polint = np.einsum('ijk, ijk->i', dl_pol, A_midpol)

    dl_tor = np.vstack((r_surf[1:, :, :] - r_surf[:-1, :, :],
                        (r_surf[0, :, :] - r_surf[-1, :, :])[None, :, :]))
    A_midtor = 0.5 * np.vstack(
        (vec_surf[1:, :, :] + vec_surf[:-1, :, :],
         (vec_surf[0, :, :] + vec_surf[-1, :, :])[None, :, :]))
    torint = np.einsum('ijk, ijk->j', dl_tor, A_midtor)
    return polint, torint


def flux_through_loop_vectorpotential(l_loop, ll, dl, I_arr):
    """
    calculate the magnetic flux through a loop given by a set of points l_loop
    by integrating the vector potential along it.
    """
    vector_potential_onloop = vmap(vector_potential, (0, None, None, None), 0)
    closedloop = np.vstack((l_loop, l_loop[0, :]))
    loop_dl = closedloop[
        1:, :] - closedloop[:-1, :]  # vector from each point to next
    A_arr = vector_potential_onloop(closedloop, ll, dl, I_arr)
    # Vector potential averaged between beginning and end of each dl segment. Is this needed?
    midpoint_A = (A_arr[:-1, :] + A_arr[1:, :]) / 2
    # dot product is summing over elementwise multiplication of axis 1, integration is sum over axis 0
    flux = np.sum(midpoint_A * loop_dl)
    return flux


def coil_force(ll, dl):
    """
    Calculate the Lorentz Force on the coils
    """
    # vector map biot_savart
    BS_coils = vmap(
        biot_savart_oncoil, (0, None, None, None), 0
    )  # map the input (which will be l) over the first dimension of the input aray (coils)
    BS_elements = vmap(
        BS_coils, (0, None, None, None), 0
    )  # map the input (which is the l array) over it's second dimension (elements of each coil).
    elementwise_force = np.cross(dl, BS_elements(ll, ll, dl, I_arr))
    #percoil_force = np.sum(elementwise_force), axis = 1) # sum over elements
    Total_force = np.sum(np.linalg.norm(elementwise_force, axis=-1),
                         axis=(0, 1))
    return Total_force


def quadratic_flux(r_surf, nn, sg, dl, ll, I_arr):
    """
    calculate the quadratic flux of the field (generated by coils specified
    by dl and l) through the surface (specified by nn and sg).
    Arguments:
    *r_surf*: array of points on the surface
    *nn*: normal to the surface in same shape as points
    *sg*: metric factor in same shape as points
    *dl*:
    *l* :
    returns:
    Quadratic flux through surface caused by the coils.
    """
    return (0.5 * np.sum(
        np.sum(nn * biot_savart_surface(r_surf, dl, ll, I_arr), axis=-1)**2 *
        sg))


def r(p, theta):
    """
    get cartesian positions from a coilSeries object through fourier
    unpacking.
    See Focus documentation for how the fourier components are packed.
    Arguments:
    *p*: (?,?,?)-array containing the sine and cosine series describing the coils
    *theta*: series of angles where the fourier series is to be evaluated.
    Returns:
    *r*: (?,?,?)-array of positions on each coil, where last element equals the first
    """
    r = np.zeros((3, p.shape[1], NS + 1))
    for m in range(p.shape[2]):
        r += p[:3, :, None, m] * np.cos(m * theta)[None, None, :] + p[
            3:, :, None, m] * np.sin(m * theta)[None, None, :]
    return np.transpose(r, (1, 2, 0))


def loss(r_surf, nn, sg, length_weight, force_weight, target_flux, p, I_arr):
    """
    Loss function that decreases the 'better' the configuration p
    gets.
    Currently given by quadratic_flux plus weight factor times coil length.
    """
    closed_l = r(p, theta)  # Theta is not functionally passed! Problem?
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    q_flux = quadratic_flux(r_surf, nn, sg, dl, ll, I_arr)
    polint, torint = flux_through_all_loops(r_surf, ll, dl, I_arr)
    fluxerr = np.linalg.norm(
        polint - target_flux)  # sum squared? divide by number of loops?
    polfluxvar = np.var(torint)

    length_err = length_weight * np.sum(np.sqrt(np.sum(dl**2, axis=-1)))
    #force = force_weight * coil_force(ll, dl)
    #oneflux = 1 * np.abs(target_flux - flux_through_loop_vectorpotential(
    #    r_surf[0, :, :], ll, dl, I_arr))
    return  q_flux + length_err + fluxerr # + polfluxerr + torfluxvar + force


#######################################################################
# JAX/Python Function Transformations
#######################################################################


#  define objective function by calling loss with specified values
@jit
def objective_function(objective_dict):
    p = objective_dict["p"]
    I_arr = objective_dict["I_arr"]
    return loss(r_surf, nn, sg, 0.0010, 0.1, -1, p, I_arr)


# vectorize (JAX).
# vmap takes the functin biot_savart and applies it to every element of an array
# whose shape is specified by the other arguments.
biot_savart_surface = vmap(vmap(biot_savart, (0, None, None, None), 0),
                           (1, None, None, None), 1)

# automatic differentiation (JAX), returns grad(f)
grad_func = grad(objective_function)

# jit-compile to CPU/GPU (JAX)
jit_grad_func = jit(grad_func)

#######################################################################
# Optimization
#######################################################################

# Now using an objective dictionary, to store multiple variables that we
# are optimizing against
objective_dict = {"p": p, "I_arr": I_arr}

print("loss is {}".format(objective_function(objective_dict)))

objective_dicts = []

for n in range(N):
    gradient = jit_grad_func(objective_dict)
    objective_dict["p"] = objective_dict["p"] - gradient["p"] * lr
    objective_dict[
        "I_arr"] = objective_dict["I_arr"] - gradient["I_arr"] *  lr
    print(n)
    print("taking a shape step of size: {}".format(np.linalg.norm(gradient["p"])*lr))
    print("taking a current step of size: {}".format(np.linalg.norm(gradient["I_arr"])*lr))
    print("loss is {}".format(objective_function(objective_dict)))
    objective_dicts.append(objective_dict.copy())

print("loss is {}".format(objective_function(objective_dict)))

with tb.open_file("coils_force.hdf5", "w") as f:
    f.create_array("/", "coilSeries", numpy.asarray(objective_dict["p"]))
    f.create_array("/", "I_arr", numpy.asarray(objective_dict["I_arr"]))

from jax.experimental.ode import odeint

from mpl_toolkits.mplot3d import Axes3D


def integrate_fieldline(startpoint, objective_dict):
    closed_l = r(objective_dict['p'], theta)
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    jitfield = jit(
        lambda r_eval, t: biot_savart(r_eval, dl, ll, objective_dict['I_arr']))
    startpos = r_surf[0, 0, :]
    times = np.linspace(0, 10, 10000)
    fieldline = odeint(jitfield, startpos, times)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(fieldline[:, 0], fieldline[:, 1], fieldline[:, 2])


from matplotlib import pyplot as plt
