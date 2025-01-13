# Spectral Derivatives
[![Build Status](https://github.com/pavelkomarov/spectral-derivatives/actions/workflows/build.yml/badge.svg)](https://github.com/pavelkomarov/spectral-derivatives/actions)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/spectral-derivatives/badge.svg?branch=main)](https://coveralls.io/github/pavelkomarov/spectral-derivatives?branch=main)

[Documentation](https://pavelkomarov.com/spectral-derivatives/specderiv.html), [How it works](https://pavelkomarov.com/spectral-derivatives/math.pdf).

This repository is the home of Python code to accomplish taking spectral derivatives, with the Chebyshev basis or the Fourier basis, based on some [pretty elegant math](https://pavelkomarov.com/spectral-derivatives/math.pdf). It's useful any time you want to take a derivative programmatically, such as for doing PDE simulations.

## Installation and Usage
The package is a single module containing derivative functions. Before installing the module you will need `numpy`, `scipy`, `scikit-learn`, and `matplotlib`. To install, execute:
```shell
python3 -m pip install spectral-derivatives
```
or from the source code
```shell
python3 -m pip install .
```
You should now be able to
```python
>>>from specderiv import *
```

For usage examples, see the Jupyter notebooks in the examples folder: [Chebyshev](https://github.com/pavelkomarov/spectral-derivatives/blob/main/examples/chebyshev.ipynb) and [Fourier](https://github.com/pavelkomarov/spectral-derivatives/blob/main/examples/fourier.ipynb)

## References

1. Trefethen, N., 2000, Spectral Methods in Matlab, https://epubs.siam.org/doi/epdf/10.1137/1.9780898719598.ch8
2. Johnson, S., 2011, Notes on FFT-based differentiation, https://math.mit.edu/ stevenj/fft-deriv.pdf
3. Kutz, J.N., 2023, Data-Driven Modeling & Scientific Computation, Ch. 11, https://faculty.washington.edu/kutz/kutz_book_v2.pdf