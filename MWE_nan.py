import jax.numpy as np
from jax import jit, grad
from jax.ops import index_update, index


def singularity_containing_integral(a, b):
    """
    approximate the integral of 1/(a*(x-b)) from zero to 10
    a: scalar
    b: integer
    """
    points = a * (np.linspace(0, 10, 1001) - b
                  )  # calculate the the denominator
    inverse = 1 / points  # invert
    inverse_cleaned = index_update(inverse, index[100*b], 0) # replace the infinity
    return np.sum(inverse_cleaned *
                  (10 / 1001))  # multiply by step size and sum


print("the integral of 1/(5*(3-x)) from 0 to 10 \
        approximately equals {}".format(singularity_containing_integral(5.,
                                                                        3)))

# gradient function w.r.t. first argument:
grad_int = grad(singularity_containing_integral, argnums=0)

print("The gradient does not compute: {}".format(grad_int(5., 3)))


def sliced_sum_array(a):
    points = 1 / np.linspace(-.1, a, 100)
    return np.sum(points[1:])


grad_slice = grad(sliced_sum_array, argnums=0)

print("the sum skipping the singularity is {}".format(sliced_sum_array(12.)))
print("the gradient skipping the singularity is {}".format(grad_slice(12.)))


def singularity_containing_integral_skip(a, b):
    """
    approximate the integral of 1/(a*(x-b)) from zero to 10
    a: scalar
    b: integer
    """
    points = a * (np.linspace(0, 10, 1001) - b
                  )  # calculate the the denominator
    inverse = points**-1  # invert
    return (10 / 1001) * (np.sum(inverse[:10 * b])
                          )  # multiply by step size and sum



def sliced_sum_array(a):
    points = 1 / np.linspace(0., a, 100)
    return points[-1]

grad_slice = grad(sliced_sum_array, argnums=0)

print("the last element of the array is {}".format(sliced_sum_array(12.)))
print("the gradient of the last element is {}".format(grad_slice(12.)))



import jax.numpy as np
from jax import jit, grad, vmap
from jax.ops import index_update, index

def sliced_sum_array(a):
    points = np.linspace(0., a, 100)
    points = points.at[0].set(1.)  # added guard *before* the reciprocal
    points = 1 / points
    points_cleaned = points.at[0].set(0.)
    return np.sum(points_cleaned[0:])

def sum_sliced_sums(input1, input2):
    mapped_sliced_sum_array = vmap(sliced_sum_array, 0, 0)
    mapped_result = mapped_sliced_sum_array(inputs)
    return sum(mapped_result)

grad_sum_sliced_sums = grad(sum_sliced_sums, argnums=0)

inputs = np.linspace(5, 10,3)


print("the sum skipping the singularity is {}".format(sum_sliced_sums(inputs)))
print("the gradient skipping the singularity is {}".format(grad_sum_sliced_sums(inputs)))
