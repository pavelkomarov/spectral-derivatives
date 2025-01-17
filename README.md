# Spectral Derivatives
[![Build Status](https://github.com/pavelkomarov/spectral-derivatives/actions/workflows/build.yml/badge.svg)](https://github.com/pavelkomarov/spectral-derivatives/actions)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/spectral-derivatives/badge.svg?branch=main)](https://coveralls.io/github/pavelkomarov/spectral-derivatives?branch=main)

[Documentation](https://pavelkomarov.com/spectral-derivatives/specderiv.html), [How it works](https://pavelkomarov.com/spectral-derivatives/math.pdf).

This repository is home to Python code that can take spectral derivatives with the Chebyshev basis or the Fourier basis, based on some [pretty elegant, deep math](https://pavelkomarov.com/spectral-derivatives/math.pdf). It's useful any time you want to take a derivative numerically, such as for doing PDE simulations.

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
>>> yx_n = np.sin(x_n) # can be periodic or aperiodic on the domain [-1, 1]
>>> dyx_n = cheb_deriv(yx_n, 1)
>>>
>>> th_n = np.arange(20) * 2*np.pi / 20 # equispaced, excludes last point
>>> yth_n = np.sin(th_n) # must be periodic on the domain [0, 2pi)
>>> dyth_n = fourier_deriv(yth_n, 1)
```
Note the `deriv` functions are agnostic to where the `y` data comes from, but for accurate results you'll need to use equispaced samples on an open periodic interval for `fourier` and cosine-spaced points for `chebyshev`. Specifically, if you want to sample from $t_0$ to $t_M$ (exclusive) or $t_N$ (inclusive), then you can convert one of the prescribed domains with:

$$[0, 2\pi) \rightarrow [0, M) \cdot (\underbrace{t_1 - t_0}_{\delta t}) + t_0$$
$$[1, -1] \rightarrow \cos(\frac{\pi [0, N]}{N}) \cdot \frac{t_N - t_0}{2} + \frac{t_N + t_0}{2}$$

For further usage examples, see the Jupyter notebooks: [Chebyshev](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/chebyshev.ipynb) and [Fourier](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/fourier.ipynb)

## References

1. Trefethen, N., 2000, Spectral Methods in Matlab, https://epubs.siam.org/doi/epdf/10.1137/1.9780898719598.ch8
2. Johnson, S., 2011, Notes on FFT-based differentiation, https://math.mit.edu/~stevenj/fft-deriv.pdf
3. Kutz, J.N., 2023, Data-Driven Modeling & Scientific Computation, Ch. 11, https://faculty.washington.edu/kutz/kutz_book_v2.pdf