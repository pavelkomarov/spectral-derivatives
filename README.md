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
If you have a function that isn't on one of these prescribed domains, you'll have to [scale](https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:transformations/x2ec2f6f830c9fb89:scale/v/scaling-functions-intro) and [shift](https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:transformations/x2ec2f6f830c9fb89:shift/v/shifting-functions-intro) your function's dependent variable(s) to fit one of these domains, use the derivative function of your choice, then scale and shift back. Tricky, but straightforward. If we're sampling the domain at $\vec{t}$, then we should be able to convert with:

$$[0, 2\pi) \rightarrow [0, M) \cdot \delta t + t_0$$
$$[1, -1] \rightarrow \cos(\frac{\pi [0, N]}{N}) \cdot \frac{t_N - t_0}{2} + \frac{t_N + t_0}{2}$$

For further usage examples, see the Jupyter notebooks in the notebooks folder: [Chebyshev](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/chebyshev.ipynb) and [Fourier](https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/fourier.ipynb)

## References

1. Trefethen, N., 2000, Spectral Methods in Matlab, https://epubs.siam.org/doi/epdf/10.1137/1.9780898719598.ch8
2. Johnson, S., 2011, Notes on FFT-based differentiation, https://math.mit.edu/~stevenj/fft-deriv.pdf
3. Kutz, J.N., 2023, Data-Driven Modeling & Scientific Computation, Ch. 11, https://faculty.washington.edu/kutz/kutz_book_v2.pdf