
def vector_potential_indexed2(coil_i, segment_j, ll, dl, I_arr):
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
    other_ll = np.zeros((19, 32, 3))
    other_ll += np.vstack((ll[:coil_i, :, :], ll[coil_i+1:, :, :]))
    other_dl = np.vstack((dl[:coil_i, :, :], dl[coil_i+1:, :, :]))
    other_I = np.hstack((I_arr[:coil_i], I_arr[coil_i+1:]))
    other_top = other_dl * other_I[:, None, None]
    other_bottom = np.linalg.norm(ll[coil_i, segment_j, :][None, None, :] - other_ll, axis=-1)
    other_vec = np.sum(other_top/other_bottom[:,:,None], axis=(0,1))

    self_ll = np.vstack((ll[coil_i, :segment_j, :], ll[coil_i, segment_j+1:, :]))
    self_dl = np.vstack((dl[coil_i, :segment_j, :], dl[coil_i, segment_j+1:, :]))
    self_I = I_arr[coil_i]
    self_top = self_I * self_dl
    self_bottom = np.linalg.norm(ll[coil_i, segment_j, :][None, :] - self_ll, axis = -1)
    self_vec = np.sum(self_top/self_bottom[:, None], axis=(0,))
    return self_vec + other_vec


def vector_potential_protected(r_eval, ll, dl, I_arr):
    """
    calculate the vector potential in the Lorentz Gauge at position r_eval,
    replacing infinites due to evaluation on the current filament with
    an expression for a finite-width tube.
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
    distances = np.linalg.norm(r_eval[None, None, :] - ll, axis=-1)
    distances_cleaned = np.where(distances<1e-5, np.ones(distances.shape), distances)  # clean zeroes before and after division to stop NaNs from propagating
    inverse_distances = 1/distances_cleaned
    inverse_cleaned = np.where(distances<1e-5, np.zeros(distances.shape), inverse_distances)
    return np.sum(top*inverse_cleaned[:,:,None], axis=(0, 1))


# map over the two input arrays, returning a n, m, 3 array?
#index_mapped_vector_potential = vmap(vmap(vector_potential_indexed, (None, 0, None, None, None)), (0, None, None, None, None))

def normvecpot(coil_i, segment_j, ll, dl, I_arr):
    return np.linalg.norm(vector_potential_indexed(coil_i, segment_j, ll, dl, I_arr))




def test_A(objective_array):
    p, I_arr = array2both(objective_array)
    #pc = p2pc(p)
    closed_l = r(p, theta)
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    coil_idx = np.array(range(ll.shape[0]))
    segment_idx = np.array(range(ll.shape[1]))
    index_mapped_vector_potential = vmap(vmap(vector_potential_indexed, (None, 0, None, None, None)), (0, None, None, None, None))
    A_on_coils = index_mapped_vector_potential(coil_idx, segment_idx, ll, dl, I_arr)
    return np.sum(A_on_coils)

def test_A(objective_array):
    p, I_arr = array2both(objective_array)
    pc = p2pc(p)
    closed_l = r_compl(pc)
    ll = closed_l[:, :-1, :]
    dl = closed_l[:, :-1, :] - closed_l[:, 1:, :]
    A_single= vector_potential_indexed(3,3, ll, dl, I_arr)
    return np.sum(A_single)

#grad(test_A)(objective_array)

def vector_potential_oncoil(r_eval, ll, dl, I_arr, tube_radius):
    """
    calculate the vector potential in the Lorentz Gauge at position r_eval,
    replacing infinites due to evaluation on the current filament with
    an expression for a finite-width tube.
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
    singular_idxs = np.where(bottom == 0.)
    oncoil_length = np.linalg.norm(dl[singular_idxs], axis=1)
    segment_potential = tube_vector_potential(oncoil_length, tube_radius)
    adjusted_bottom = index_update(bottom, singular_idxs, segment_potential)
    return top / adjusted_bottom[:, :, None]



@jit
def objective_dictfun(objective_dict):
    return loss(r_surf, nn, sg, 0.010, 0.1, -1, objective_dict["p"],
                objective_dict["I_arr"])


def make_convergence_movie():
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
    def callbackF(objective_array):
        global N
        if N%5==0:
            print("making a figure!")
            mlab.clf()
            p, I_arr = array2both(objective_array)
            ll = r(p, cfg.theta)
            draw_coils(ll)
            draw_surface(cfg.r_surf)
            mlab.savefig('convergence_{}.png'.format(N), size = (1000,1000))
        N+=1
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
    N=0
    res = optimize.minimize(objective, objective_array, jac=jit_grad_func, callback=callbackF)
