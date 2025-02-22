# Spectral Derivatives
[![Build Status](https://github.com/pavelkomarov/spectral-derivatives/actions/workflows/build.yml/badge.svg)](https://github.com/pavelkomarov/spectral-derivatives/actions)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/spectral-derivatives/badge.svg?branch=main)](https://coveralls.io/github/pavelkomarov/spectral-derivatives?branch=main)

[Documentation](https://pavelkomarov.com/spectral-derivatives/specderiv.html), [How it works](https://pavelkomarov.com/spectral-derivatives/math.pdf).

This repository is home to Python code that can take spectral derivatives with the Chebyshev basis or the Fourier basis, based on some [pretty elegant, deep math](https://pavelkomarov.com/spectral-derivatives/math.pdf). That is, given a vector representing samples of a smooth function, the code returns a numerical derivative, indicating how fast an interpolation built of the basis functions is changing at the sample points.

When using the Fourier basis, spectral derivatives require periodic boundaries, but the Chebyshev basis allows arbitrary boundaries, extending the method to a much wider class of functions.

This package can be useful any time you want to take a derivative numerically, such as for doing PDE simulations. For taking derivatives of noisy data, spectral methods naturally enable filtering by frequency.

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
For further usage examples, including in higher dimension, see the Jupyter notebooks: [Chebyshev](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/chebyshev.ipynb) and [Fourier](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/fourier.ipynb).

Note that for accurate results you'll need to use equispaced samples on an open periodic interval for `fourier` and cosine-spaced points for `chebyshev`. For examples which use arbitrary domains, see [this notebook](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/arbitrary_domains.ipynb).

## References

1. Trefethen, N., 2000, Spectral Methods in Matlab, https://epubs.siam.org/doi/epdf/10.1137/1.9780898719598.ch8
2. Johnson, S., 2011, Notes on FFT-based differentiation, https://math.mit.edu/~stevenj/fft-deriv.pdf
3. Kutz, J.N., 2023, Data-Driven Modeling & Scientific Computation, Ch. 11, https://faculty.washington.edu/kutz/kutz_book_v2.pdf