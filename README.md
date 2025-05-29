# Spectral Derivatives
[![Build Status](https://github.com/pavelkomarov/spectral-derivatives/actions/workflows/build.yml/badge.svg)](https://github.com/pavelkomarov/spectral-derivatives/actions)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/spectral-derivatives/badge.svg?branch=main)](https://coveralls.io/github/pavelkomarov/spectral-derivatives?branch=main)

[Documentation](https://pavelkomarov.com/spectral-derivatives/specderiv.html), [How it works](https://pavelkomarov.com/spectral-derivatives/math.pdf).

This repository is home to Python code that can take spectral derivatives using the [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev_polynomials), [Fourier](https://mathworld.wolfram.com/FourierSeries.html), [Legendre](https://en.wikipedia.org/wiki/Legendre_polynomials), and [Bernstein](https://en.wikipedia.org/wiki/Bernstein_polynomial) bases, grounded in some [pretty elegant, deep math](https://pavelkomarov.com/spectral-derivatives/math.pdf). That is, given a vector representing samples of a smooth function, the code returns numerical derivatives, indicating slope, curvature, etc. at the sample points of an interpolation built from the basis functions.

When using the Fourier basis, spectral derivatives require periodic boundaries, but the polynomial bases (fastest to fit being the Chebyshev basis) allow arbitrary boundaries, extending the method to a much wider class of functions (albeit best with nonuniform sampling).

This package can be useful any time you want to take a derivative numerically, such as for doing PDE simulations. For taking derivatives of noisy data, spectral methods naturally enable global filtering by weighting basis function contributions, although polynomial bases suffer [some weakness](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/filtering_noise.ipynb).

Note these methods are best for situations when you *don't know the generating function of your data*; if the generator is known, then autodifferentiation tools like [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) are your friend.

## Installation and Usage
The package is a single module containing derivative functions. To install, execute:
```shell
python3 -m pip install spectral-derivatives
```
or from the source code
```shell
python3 -m pip install .
```
You should now be able to
```python
>>> from specderiv import *
>>> import numpy as np
>>>
>>> x_n = np.cos(np.arange(21) * np.pi / 20) # cosine-spaced, includes last point
>>> y_n = np.sin(x_n) # can be periodic or aperiodic on domain [a, b]
>>> dy_n = cheb_deriv(y_n, x_n, 1)
>>>
>>> th_n = np.arange(20) * 2*np.pi / 20 # equispaced, excludes last point
>>> y_n = np.sin(th_n) # must be periodic on domain [a, b)
>>> dy_n = fourier_deriv(y_n, th_n, 1)
```
For further usage examples, including in higher dimension, see the Jupyter notebooks: [Chebyshev](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/chebyshev.ipynb), [Fourier](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/fourier.ipynb), [Legendre](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/legendre.ipynb), and [Bernstein](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/bernstein.ipynb).

Note that for fastest and most accurate results you should use equispaced samples on an open periodic interval with `fourier_deriv` and cosine-spaced points with `cheb_deriv`. `legendre_deriv` and `bern_deriv` are slower. All methods support [arbitrary domains](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/arbitrary_domains.ipynb) by internally performing an affine transformation to some canonical domain. And all methods support differentiating multidimensional data along an `axis`.

## References

1. Trefethen, N., 2000, Spectral Methods in Matlab, https://epubs.siam.org/doi/epdf/10.1137/1.9780898719598.ch8
2. Johnson, S., 2011, Notes on FFT-based differentiation, https://math.mit.edu/~stevenj/fft-deriv.pdf
3. Kutz, J.N., 2023, Data-Driven Modeling & Scientific Computation, Ch. 11, https://faculty.washington.edu/kutz/kutz_book_v2.pdf
4. Breuer, K. & Everson, R., 1990, On the errors incurred calculating derivatives using Chebyshev polynomials,
https://doi.org/10.1016/0021-9991(92)90274-3
