import numpy as np
import numba

# Adapted from https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
# Goal is to implement a numba compatible polyfit (note does not include error handling)

@numba.njit
def fit_poly(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    a: np.ndarray = _coeff_mat(x, deg)
    p: np.ndarray = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]


# Define Functions Using Numba
# Idea here is to solve ax = b, using least squares, where a represents our coefficients e.g. x**2, x, constants
@numba.njit
def _coeff_mat(x: np.ndarray, deg: int) -> np.ndarray:
    mat_: np.ndarray = np.zeros(shape=(x.shape[0], deg + 1))
    const: np.ndarray = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x ** n
    return mat_


@numba.njit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


if __name__ == '__main__':
    # Create Dummy Data and use existing numpy polyfit as test
    x = np.linspace(0, 2, 20)
    y = np.cos(x) + 0.3 * np.random.rand(20)

    import time
    t0 = time.time()
    for _ in range(1000):
        coeffs_numpy = np.polyfit(x, y, 3)
    print("1000 x numpy:", time.time()-t0)

    # force compile numba
    coeffs_numba = fit_poly(x, y, deg=3)

    t0 = time.time()
    for _ in range(1000):
        coeffs_numba = fit_poly(x, y, deg=3)
    print("1000 x numba:", time.time() - t0)

    assert np.allclose(coeffs_numba, coeffs_numpy)
