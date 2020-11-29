import jax.numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import h5py
from jax import jit, grad, vmap
from objective_function import config as cfg

from jax.config import config
import tables as tb
from functools import partial
import numpy as numpy
import scipy.optimize as optimize
from jax.ops import index_update, index
#config.update("jax_enable_x64", True)



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


def flux_through_poloidal_loops(r_surf, ll, dl, I_arr):
    """
    returns the magnetic flux through all poloidal and toroidal loops
    of the surface array.
    Arguments:
    *r_surf*: (n, m, 3) array of cartesian coordinates on the surface.
    [:,0,:] must be the first poloidal loop, and [0,:,:] the first toroidal loop.
    *dl*: ( n_coils, nsegments, 3)-array along the coil segments.
    other coil line segment
    *ll* ( n_coils, nsegments, 3)-array of the position of each coil segment
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
         (vec_surf[:, 0, :] + vec_surf[:, -1, :]
          )[:, None, :]))  # why not evaluate A at mid-grid locations?
    # sum over axis 0 and 2 only, the product of the two. Calculates the inner product, and then "integrates" poloidally.
    polint = np.einsum('ijk, ijk->i', dl_pol, A_midpol)
    return polint




def flux_through_all_loops(r_surf, ll, dl, I_arr):
    """
    returns the magnetic flux through all poloidal and toroidal loops
    of the surface array.
    Arguments:
    *r_surf*: (n, m, 3) array of cartesian coordinates on the surface.
    [:,0,:] must be the first poloidal loop, and [0,:,:] the first toroidal loop.
    *dl*: ( n_coils, nsegments, 3)-array along the coil segments.
    other coil line segment
    *ll* ( n_coils, nsegments, 3)-array of the position of each coil segment
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
         (vec_surf[:, 0, :] + vec_surf[:, -1, :]
          )[:, None, :]))  # why not evaluate A at mid-grid locations?
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


def coil_force(ll, dl, lltangent):
    """
    Calculate the Lorentz Force on the coils
    args:
    other coil line segment
    *ll* ( n_coils, nsegments, 3)-array of the position of each coil segment
    *dl*: ( n_coils, nsegments, 3)-array along the coil segments.
    *lltangent*: tangent vector to coil at the ll positions (drdtheta function)
    """
    # vector map biot_savart
    BS_coils = vmap(
        biot_savart_oncoil, (0, None, None, None), 0
    )  # map the input (which will be l) over the first dimension of the input aray (coils)
    BS_elements = vmap(
        BS_coils, (0, None, None, None), 0
    )  # map the input (which is the l array) over it's second dimension (elements of each coil).
    elementwise_force = np.cross(lltangent, BS_elements(ll, ll, dl, I_arr))
    return elementwise_force




def p2pc(p):
    "make the real sine and cosine coefficients into complex array"
    return np.complex128(p[:3, :, :] + 1j * p[3:, :, :])


def r_compl(pc):
    """
    compute position array using complex fast fourier transform
    args:
    *p*: (3, n_coils, NF) array of complex fourier components
    *NS*: number of segments
    """
    #pad pc to the proper length
    paddedpc = np.pad(pc, ((0, 0), (0, 0), (0, pad_length)), constant_values=0)
    #fourier transform and take the real part
    fourier = np.real(np.fft.fftn(paddedpc, axes=(-1, )))
    # todo: add the last element equal to the first (for dl computation)
    closed = np.dstack((fourier, fourier[:, :, 0]))
    return np.transpose(closed, (1, 2, 0))


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


def r2p(r, theta):
    """
    undo the r-function, get p back from an r
    (what about the endpoint? Set as an FFT?)
    """
    p = np.zeros(6, r.shape[1], NF)


def drdtheta(p, theta):
    """
    Analtical derivative of the coils with respect to theta.
    """
    drdtheta = np.zeros((3, p.shape[1], cfg.NS + 1))
    for m in range(p.shape[2]):
        drdtheta += - m * p[:3, :, None, m] * np.sin(m * theta)[None, None, :] + \
             m *p[3:, :, None, m] * np.cos(m * theta)[None, None, :]
    return np.transpose(drdtheta, (1, 2, 0))




def tube_vector_potential(segment_length, tube_radius):
    """
    calculate the vecto potential in the center of a coil segment
    assuming a cylindrical surface current on a tube of radius
    tube_radius, with the evaluation point on the center of the
    cylinder.
    Can be seen as the average distance of the tube walls to the
    centermost point of the tube, and a reasonable approximation
    of the vector potential on that centerline.
    Used to replace the infinity encountered in the Lorentz gauge on
    a infinitely thin cylinder.
    Assuming:
    $A= I_0 * \int_{-|dl|/2}^{|dl|/2} 1/(\sqrt{h^2+w^2})
    = I_0 \log(\frac{|dl|(\sqrt{.5 |dl|^2+w^2}+.5|dl|^2)}{w^2})
    where I_0 is taken out of the integral because it is constant.
    this function only returns the result of the log calculation.
    args:
    *segment_length*: the length of the segment where the integral
    is to be replaced
    *tube_radius*: the radius of the tube on which the surface current
    is calculated.
    """
    log_numerator = segment_length * (np.sqrt(.5 * segment_length**2 + tube_radius**2)\
                     + .5*segment_length)
    segment_potential = np.log((log_numerator/(tube_radius**2)) + 1.)
    return segment_potential



def vector_potential_indexed(coil_i, segment_j, ll, dl, I_arr):
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
    distances = ll[coil_i, segment_j, :][None, None, :] - ll
    distances_cleaned = distances.at[coil_i, segment_j, :].set(1)#remove zero
    bottom = np.sqrt(np.sum(distances_cleaned**2, axis=-1))
    #bottom_cleaned = bottom.at[coil_i, :].set(1) #set the zero value to something before division
    bottom_inv = 1/bottom
    segment_length = np.linalg.norm(dl[coil_i, segment_j, :])
    replacement_value = 1/tube_vector_potential(segment_length, .01)
    bottom_inv_cleaned = bottom_inv.at[coil_i, segment_j].set(replacement_value)
    #bottom_cleaned = index_update(bottom_inv, [coil_i, segment_j], 0)
    #A = np.sum(top / bottom_cleaned[:, :, None], axis=(0, 1))
    integrand = top*bottom_inv_cleaned[:,:,None]
    A = np.sum(integrand, axis=(0, 1))
    return A

def quadratic_flux(r_surf, nn, sg, dl, ll, I_arr):
    """
    calculate the quadratic flux of the field (generated by coils specified
    by dl and l) through the surface (specified by nn and sg).
    Arguments:
    *r_surf*: array of points on the surface
    *nn*: normal to the surface in same shape as points
    *sg*: metric factor in same shape as points
    *dl*: tangent vectors along coils
    *ll*: positons of coil segments
    returns:
    Quadratic flux through surface caused by the coils.
    """
    return (0.5 * np.sum(
        np.sum(nn * biot_savart_surface(r_surf, dl, ll, I_arr), axis=-1)**2 *
        sg))

def total_energy(objective_array):
    """
    Calculate a proxy for the energy in the configuration of
    coils.
    Ignores the infinite energy of filamentary coils by setting
    the self-contribution of the biot-savart like vector potential
    integral (vector potential induced by the current in the coil
    segment on which we are evaluating) to zero. (this contribution
    becomes finite for a finite-build coil, but importantly it doesn't
    change with the position of the segment, so the derivative of this
    array is still good).
    """
    p, I_arr = array2both(objective_array)
    #pc = p2pc(p)
    closed_l = r(p, cfg.theta)
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    midpointgrid = 0.5 * np.hstack((ll[:, :-1, :] + ll[:, 1:, :],
                                    (ll[:, 0, :] + ll[:, -1, :])[:, None, :]))
    coil_idx = np.array(range(ll.shape[0]))
    segment_idx = np.array(range(ll.shape[1]))
    index_mapped_vector_potential = vmap(vmap(vector_potential_indexed, (None, 0, None, None, None), 0), (0, None, None, None, None), 0)
    A_on_coils = index_mapped_vector_potential(coil_idx, segment_idx, midpointgrid, dl, I_arr)
    # inner product is elementwise multiplication, summed over axis -1, result is just summing:
    coil_fluxes = np.sum(A_on_coils * dl, axis = (1,2))
    energy = np.sum(I_arr * coil_fluxes)
    return energy



def quadratic_flux_error(objective_array):
    """
    Calculate the quadratic flux through the surface as one of the optimization parameters
    """
    p, I_arr = array2both(objective_array)
    #pc = p2pc(p)
    closed_l = r(p, cfg.theta)
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    q_flux = quadratic_flux(cfg.r_surf, cfg.nn, cfg.sg, dl, ll, I_arr)
    return q_flux


def toroidal_flux(objective_array, target_flux):
    """
    Calculate the flux through each poloidal loop on the target surface,
    and return the difference with the given target flux
    """
    p, I_arr = array2both(objective_array)
    closed_l = r(p, cfg.theta)  # loop where last element equals first
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    polint = flux_through_poloidal_loops(cfg.r_surf, ll, dl, I_arr)
    fluxerr = np.linalg.norm(
        polint - target_flux)  # sum squared? divide by number of loops?
    return fluxerr


def coil_length(objective_array):
    p = array2p(objective_array)
    closed_l = r(p, cfg.theta)  # loop where last element equals first
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    total_length = np.sum(np.sqrt(np.sum(dl**2, axis=-1)))
    return total_length



def loss(r_surf, nn, sg, length_weight, force_weight, target_flux, p, I_arr):
    """
    Loss function that decreases the 'better' the configuration p
    gets.
    Currently given by quadratic_flux plus weight factor times coil length.
    """
    closed_l = r(p, cfg.theta)  # loop where last element equals first
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    q_flux = quadratic_flux(r_surf, nn, sg, dl, ll, I_arr)
    polint, torint = flux_through_all_loops(r_surf, ll, dl, I_arr)
    fluxerr = .1 * np.linalg.norm(
        polint - target_flux)  # sum squared? divide by number of loops?
    polfluxvar = np.var(torint)
    length_err = length_weight * np.sum(np.sqrt(np.sum(dl**2, axis=-1)))
    #force = force_weight * coil_force(ll, dl)
    #oneflux = 1 * np.abs(target_flux - flux_through_loop_vectorpotential(
    #    r_surf[0, :, :], ll, dl, I_arr))
    return length_err + fluxerr + q_flux  #+ polfluxvar#+ q_flux + torfluxvar + force


#######################################################################
# JAX/Python Function Transformations
#######################################################################


#  define objective function by calling loss with specified values
def objective_function(objective_array, qq, ee, ll, tt):
    """
    return the weighted sum of the optimization objectives.
    Arguments:
    *qq*: weight for the quadratic flux error (see: quadratic_flux_error)
    *ee*: Weight for the total energy function (see: total_energy)
    *ll*: weight for the coil length (see: coil_lenght)
    *tt*: weight for the toroidal flux through each toroidal loop
          (see: toroidal_flux)
    *rr*: [NOT IMPLEMENTED] weight for the coil-coil repulsion function
    """
    Q = qq * quadratic_flux_error(objective_array)
    E = ee * total_energy(objective_array)
    L = ll * coil_length(objective_array)
    T = tt * toroidal_flux(objective_array, 1.0)
    return Q+E+L+T

def objective_function2(objective_array):
    """
    return the weighted sum of the optimization objectives.
    Arguments:
    *qq*: weight for the quadratic flux error (see: quadratic_flux_error)
    *ee*: Weight for the total energy function (see: total_energy)
    *ll*: weight for the coil length (see: coil_lenght)
    *tt*: weight for the toroidal flux through each toroidal loop
          (see: toroidal_flux)
    *rr*: [NOT IMPLEMENTED] weight for the coil-coil repulsion function
    """
    Q = 1.0 * quadratic_flux_error(objective_array)
    E = .005 * total_energy(objective_array)
    L = 0.1 * coil_length(objective_array)
    T = 1.0 * toroidal_flux(objective_array, 1.0)
    return Q+E+L+T




q_flux_gradient = jit(grad(quadratic_flux_error))
energy_gradient = jit(grad(total_energy))
length_gradient = jit(grad(coil_length))
tor_flux_gradient = jit(grad(toroidal_flux, argnums=0), static_argnums=1)


def objective_and_gradient(qq, ee, ll, tt):
    """
    return the jitted objective function with the given weights
    and the jitted gradient of that function.
    Arguments:
    *qq*: weight for the quadratic flux error (see: quadratic_flux_error)
    *ee*: Weight for the total energy function (see: total_energy)
    *ll*: weight for the coil length (see: coil_lenght)
    *tt*: weight for the toroidal flux through each toroidal loop
          (see: toroidal_flux)
    *rr*: [NOT IMPLEMENTED] weight for the coil-coil repulsion function
    """
    @jit
    def ofn(objective_array):
        return objective_function(objective_array, qq, ee, ll, tt)
    gofn = jit(grad(ofn))
    return ofn, gofn


@jit
def gradient_function(objective_array):
    Q = 1.0 * q_flux_gradient(objective_array)
    E = .01 * energy_gradient(objective_array)
    L = .1 * length_gradient(objective_array)
    T = 1.0 * tor_flux_gradient(objective_array, 1.0)
    return Q+E+L+T


def array2p(objective_array):
    return np.reshape(objective_array[:cfg.p_size], cfg.p_shape)


def array2I_arr(objective_array):
    return np.reshape(objective_array[-cfg.I_arr_size:], cfg.I_arr_shape)


def array2both(objective_array):
    return (array2p(objective_array), array2I_arr(objective_array))


def make_array(p, I_arr):
    return np.append(p, I_arr)


# vectorize (JAX).
# vmap takes the functin biot_savart and applies it to every element of an array
# whose shape is specified by the other arguments.
biot_savart_surface = vmap(vmap(biot_savart, (0, None, None, None), 0),
                           (1, None, None, None), 1)


