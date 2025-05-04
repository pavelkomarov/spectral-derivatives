import numpy as np
from numpy.polynomial.chebyshev import chebder, chebfit, chebval
from scipy.fft import dct
from warnings import warn, catch_warnings, simplefilter


def cheb_deriv(y_n: np.ndarray, t_n: np.ndarray, order: int, axis: int=0, filter: callable=None):
	"""Evaluate derivatives with Chebyshev polynomials via series derivative rule.

	Args:
		y_n (np.ndarray): one-or-multi-dimensional array, values of a function, sampled at cosine-spaced points in the dimension
			of differentiation.
		t_n (np.ndarray): 1D array, where the function :math:`y` is sampled in the dimension of differentation. Use cosine-spaced
			points, i.e. :code:`t_n = x_n * (b - a)/2 + (b + a)/2` for a domain between :math:`a` and :math:`b`, where
			:code:`x_n = np.cos(np.arange(N+1) * np.pi / N)`, to enable :math:`O(N \\log N)` transforms to and from the basis domain.
			Note the ordering of these points counts *up* in :math:`n`, which is right-to-left, from 1 to -1 in the :math:`t` domain.
			If instead you want to use arbitrary sample points, this is allowed, but the code will warn that you are incurring
			:math:`O(N^3)` cost. Note both endpoints are *inclusive*.
		order (int): The order of differentiation, also called :math:`\\nu`. Must be :math:`\\geq 1`.
		axis (int, optional): For multi-dimensional :code:`y_n`, the dimension along which to take the derivative. Defaults to the
			first dimension (axis=0).
		filter (callable, optional): A function or :code:`lambda` that takes the 1D array of Chebyshev polynomial numbers, :math:`k
			= [0, ... N]`, and returns a same-shaped array of weights, which get multiplied in to the initial frequency transform of
			the data, :math:`Y_k`. Can be helpful when taking derivatives of noisy data. The default is to apply #nofilter.

	:returns: (*np.ndarray*) -- :code:`dy_n`, shaped like :code:`y_n`, samples of the :math:`\\nu^{th}` derivative of the function
	"""
	if order < 1: # allow antiderivatives with numpy's chebint? The trouble is those extra zeros [0, ... coefs]. How do you evaluate that at only N+1 points efficiently?
		raise ValueError("derivative order, nu, should be >= 1")
	if len(t_n.shape) > 1 or t_n.shape[0] != y_n.shape[axis]:
		raise ValueError("t_n should be 1D and have the same length as y_n along the axis of differentiation")
	if not (np.all(np.diff(t_n) > 0) or np.all(np.diff(t_n) < 0)):
		raise ValueError("t_n should be monotonically and strictly increasing or decreasing")
	
	first = [slice(None) for dim in y_n.shape]; first[axis] = 0; first = tuple(first) # for accessing different parts of data
	last = [slice(None) for dim in y_n.shape]; last[axis] = -1; last = tuple(last)
	s = [np.newaxis for dim in y_n.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	N = y_n.shape[axis] - 1 # We only have to care about the number of points in the dimension we're differentiating
	a, b = (t_n[0], t_n[-1]) if t_n[0] < t_n[-1] else (t_n[-1], t_n[0])
	scale = (b - a)/2; offset = (b + a)/2 # Trying to be helpful, because sampling is tricky to get right
	cosine_spaced = np.allclose(t_n, np.cos(np.arange(N+1) * np.pi/N) * scale + offset, atol=1e-5) # enables efficient frequency transforms
	
	if cosine_spaced: # Chebyshev points found, enables DCT-I!
		Y_k = dct(y_n, 1, axis=axis)/N # Transform to frequency domain using the 1st definition of the discrete cosine transform
		# In the IDCT-I we have y_n = 1/2N (Y_0 + (-1)^n Y_N + 2 \sum_{k=1}^{N-1} cos(pi k n/N) Y_k), but in the Chebyshev
		# expansion we have: y_n = \sum_{k=0}^{N} cos(pi k n/N) a_k. So we need to do some scaling to make Y_k = a_k.
		for k in (first, last): Y_k[k] /= 2
	else:
		warn("""Your function is not sampled for the DCT-I, i.e. `t_n = np.cos(np.arange(N+1) * np.pi / N) * (b - a)/2 + (b + a)/2`.
			`cheb_deriv` using chebfit() and chebval() under the hood, which costs O(N^3) instead of O(N log N).""")
		x_n = (t_n - offset)/scale # We have to work in the domain [-1, 1]
		Y_k = np.apply_along_axis(lambda v: chebfit(x_n, v, N), axis, y_n) # O(N^3) to find each fit

	if filter: Y_k *= filter(np.arange(N+1))[s]
	dY_k = chebder(Y_k, m=order, axis=axis)

	if cosine_spaced: # prepare the coefficients for an IDCT-I and then carry it out to evaluate dy_n
		z_shape = list(dY_k.shape); z_shape[axis] = order # add a block of zeros to the end of the coefficients, so we keep same number of points
		dY_k = np.concatenate((dY_k, np.zeros(z_shape)), axis=axis)
		dY_k[first] *= 2 # We should technically also scale dY_k[last] by 2, but this entry is full of 0s. We should also have to scale up all
		dy_n = dct(dY_k, 1, axis=axis)/2 # coefs by N to get the DCT coefs, but this just cancels the 2N in the denominator of the IDCT
	else: # evaluate Chebyshev functions themselves, O(N^2) for each one
		dy_n = np.apply_along_axis(lambda v: chebval(x_n, v), axis, dY_k)

	# The above is agnostic to where the data came from, pretends it came from the domain [-1, 1], but the data may actually be
	return dy_n/scale**order # smooshed from some other domain. So scale the derivative by the relative size of the t and x intervals.


def fourier_deriv(y_n: np.ndarray, t_n: np.ndarray, order: int, axis: int=0, filter: callable=None):
	"""Evaluate derivatives with complex exponentials via FFT. Caveats:

	- Only for use with periodic functions.
	- Taking the 1st derivative twice with a discrete method like this is not exactly the same as taking the second derivative.
 
	Args:
		y_n (np.ndarray): one-or-multi-dimensional array, values of a period of a periodic function, sampled at equispaced points
			in the dimension of differentiation.
		t_n (np.ndarray): 1D array, where the function :math:`y` is sampled in the dimension of differentiation. If you're using
			canonical Fourier points, this will be :code:`th_n = np.arange(M) * 2*np.pi / M` (:math:`\\theta \\in [0, 2\\pi)`). If
			you're sampling on a domain from :math:`a` to :math:`b`, this needs to be :code:`t_n = np.arange(0, M)/M * (b - a) + a`.
			Note the lower, left bound is *inclusive* and the upper, right bound is *exclusive*.
		order (int): The order of differentiation, also called :math:`\\nu`. Can be positive (derivative) or negative
			(antiderivative, raises warning).
		axis (int, optional): For multi-dimensional :code:`y_n`, the dimension along which to take the derivative. Defaults to the
			first dimension (axis=0).
		filter (callable, optional): A function or :code:`lambda` that takes the array of wavenumbers, :math:`k = [0, ...
			\\frac{M}{2} , -\\frac{M}{2} + 1, ... -1]` for even :math:`M` or :math:`k = [0, ... \\lfloor \\frac{M}{2} \\rfloor,
			-\\lfloor \\frac{M}{2} \\rfloor, ... -1]` for odd :math:`M`, and returns a same-shaped array of weights, which get
			multiplied in to the initial frequency transform of the data, :math:`Y_k`. Can be helpful when taking derivatives
			of noisy data. The default is to apply #nofilter.

	:returns: (*np.ndarray*) -- :code:`dy_n`, shaped like :code:`y_n`, samples of the :math:`\\nu^{th}` derivative of the function
	"""
	if len(t_n.shape) > 1 or t_n.shape[0] != y_n.shape[axis]:
		raise ValueError("t_n should be 1D and have the same length as y_n along the axis of differentiation")
	delta_t = np.diff(t_n)
	if not (np.all(delta_t > 0) and np.allclose(delta_t, delta_t[0])):
		raise ValueError("The domain, t_n, needs to be equispaced, ordered low-to-high on [a, ... b). Try sampling with `np.arange(0, M)/M * (b - a) + a`")

	M = y_n.shape[axis]
	# if M has an even length, then we make k = [0, 1, ... M/2 - 1, 0 or M/2, -M/2 + 1, ... -1]
	# if M has odd length, k = [0, 1, ... floor(M/2), -floor(M/2), ... -1]
	k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))
	if M % 2 == 0 and order % 2 == 1: k[M//2] = 0 # odd derivatives get the Nyquist element zeroed out, if there is one

	s = [np.newaxis for dim in y_n.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y_k = np.fft.fft(y_n, axis=axis)
	if filter: Y_k *= filter(k)[s]
	with catch_warnings(): simplefilter("ignore", category=RuntimeWarning); Y_nu = (1j * k[s])**order * Y_k # if order < 0, we're dividing by 0 at k=0
	if order < 0: Y_nu[np.where(k==0)] = 0; warn("+c information lost in antiderivative") # Get rid of NaNs. Enables taking the antiderivative.
	dy_n = np.fft.ifft(Y_nu, axis=axis).real if not np.iscomplexobj(y_n) else np.fft.ifft(Y_nu, axis=axis)

	# The above is agnostic to where the data came from, pretends it came from the domain [0, 2pi), but the data may actually
	scale = (t_n[M-1] + t_n[1] - 2*t_n[0])/(2*np.pi) # be smooshed from some other domain. So scale the derivative by the
	return dy_n/scale**order 						# relative size of the t and theta intervals.
