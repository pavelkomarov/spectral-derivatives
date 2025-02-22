import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.fft import dct, dst
from collections import deque
from warnings import warn, catch_warnings, simplefilter


def cheb_deriv(y_n: np.ndarray, t_n: np.ndarray, order: int, axis: int=0, filter: callable=None, dct_type=1, calc_endpoints=True):
	"""Evaluate derivatives with Chebyshev polynomials via discrete cosine and sine transforms. Caveats:

	- Taking the 1st derivative twice with a discrete method like this is not exactly the same as taking the second derivative.
	- For derivatives over the 4th, this method presently returns :code:`NaN` at the edges of the domain. Be cautious if passing
	  the result to another function.

	Args:
		y_n (np.ndarray): one-or-multi-dimensional array, values of a function, sampled at cosine-spaced points in the dimension
			of differentiation.
		t_n (np.ndarray): 1D array, where the function :math:`y` is sampled in the dimension of differentation. If you're using
			canonical Chebyshev points, this will be :code:`x_n = np.cos(np.arange(N+1) * np.pi / N)` (:math:`x \\in [1, -1]`).
			If you're sampling on a domain from :math:`a` to :math:`b`, this needs to be :code:`t_n = np.cos(np.arange(N+1) *
			np.pi / N) * (b - a)/2 + (b + a)/2`. Note the order is high-to-low in the :math:`x` or :math:`t` domain, but low-to-high
			in :math:`n`. Also note both endpoints are *inclusive*.
		order (int): The order of differentiation, also called :math:`\\nu`. Must be :math:`\\geq 1`.
		axis (int, optional): For multi-dimensional :code:`y_n`, the dimension along which to take the derivative. Defaults to the
			first dimension (axis=0).
		filter (callable, optional): A function or :code:`lambda` that takes the 1D array of wavenumbers, :math:`k = [0, ... N]`,
			and returns a same-shaped array of weights, which get multiplied in to the initial frequency transform of the data,
			:math:`Y_k`. Can be helpful when taking derivatives of noisy data. The default is to apply #nofilter.
		dct_type (int, optional): 1 or 2, whether to use DCT-I or DCT-II. Defaults to DCT-I.
		calc_endpoints (bool, optional): Whether to calculate the endpoints of the answer, in case they are unnecessary for a
			particular use case. Defaults to True. False silences the NaN warning for order > 4.
 
	:returns: (*np.ndarray*) -- :code:`dy_n`, shaped like :code:`y_n`, samples of the :math:`\\nu^{th}` derivative of the function
	"""
	# We only have to care about the number of points in the dimension we're differentiating
	N = y_n.shape[axis] - 1 if dct_type == 1 else y_n.shape[axis] - 3 # if type is 1, we count [0, ... N], if type 2, the endpoints are tacked on additionally
	M = 2*N if dct_type == 1 else 2*(N+1) # Normalization factor is larger for DCT-II based on repeats of endpoints in equivalent FFT
	x_n = np.cos(np.arange(N+1) * np.pi/N) if dct_type == 1 else np.concatenate(([1], np.cos((np.arange(N+1) + 0.5) * np.pi/(N+1)), [-1])) # canonical sampling

	if order < 1:
		raise ValueError("derivative order, nu, should be >= 1")
	if dct_type not in (1, 2):
		raise ValueError("DCT type must be 1 or 2")
	if len(t_n.shape) > 1 or t_n.shape[0] != y_n.shape[axis]:
		raise ValueError("t_n should be 1D and have the same length as y_n along the axis of differentiation")
	if not np.all(np.diff(t_n) < 0):
		raise ValueError("The domain, t_n, should be ordered high-to-low, [b, ... a]. Try sampling with `np.cos(np.arange(N+1) * np.pi / N) * (b - a)/2 + (b + a)/2`")
	scale = (t_n[0] - t_n[-1])/2; offset = (t_n[0] + t_n[-1])/2 # Trying to be helpful, because sampling is tricky to get right
	if not np.allclose(t_n, x_n * scale + offset, atol=1e-5):
		raise ValueError(f"""Your function is not sampled appropriately for the DCT-{'I'*dct_type} Try sampling with
			{'`np.cos(np.arange(N+1) * np.pi / N) * (b - a)/2 + (b + a)/2`' if dct_type == 1 else
			'`np.concatenate(([b], np.cos((np.arange(N+1) + 0.5) * np.pi/(N+1)) * (b - a)/2 + (b + a)/2, [a]))'}""")

	first = [slice(None) for dim in y_n.shape]; first[axis] = 0; first = tuple(first) # for accessing different parts of data
	last = [slice(None) for dim in y_n.shape]; last[axis] = y_n.shape[axis] - 1; last = tuple(last)
	middle = [slice(None) for dim in y_n.shape]; middle[axis] = slice(1, -1); middle = tuple(middle)
	s = [np.newaxis for dim in y_n.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y_k = dct(y_n, 1, axis=axis) if dct_type == 1 else dct(y_n[middle], 2, axis=axis) # Transform to frequency domain using the 1st definition of the discrete cosine transform
	k = np.arange(N+1) # [0, ... N], Chebyshev basis polynomial (in x)/wavenumber (in theta) iterator
	if filter: Y_k *= filter(k)[s]

	y_primes = [] # Store all derivatives in theta up to the nu^th, because we need them all for reconstruction.
	for mu in range(1, order + 1):
		Y_mu = (1j * k[s])**mu * Y_k
		if mu % 2: # odd derivative
			# In DST-I case Y_mu[k=0 and N] = 0 and so are not needed for the DST, so only pass the [middle] entries
			# In DST-III case, Y_mu[0 and N+1] = 0. roll() shifts to the left, so Y'_0 is treated like Y'_{N+1}, and we pass in starting at k=1
			y_primes.append(dst(1j * Y_mu[middle], 1, axis=axis).real / M if dct_type == 1 # d/dtheta y = the inverse transform of DST-1 = 1/M * DST-1. Extra j for equivalence with IFFT.
				else dst(1j * np.roll(Y_mu, -1), 3, axis=axis).real / M) # inverse of DST-II is 1/M * DST-III. Im{y_prime} = 0 for real y, so just keep real.
		else: # even derivative
			y_primes.append(dct(Y_mu, 1, axis=axis)[middle].real / M if dct_type == 1 # the inverse transform of DCT-1 is 1/M * DCT-1. Slice off ends to get same length as DST-I result.
				else dct(Y_mu, 3, axis=axis).real / M) # inverse of DCT-II is 1/M * DCT-III. Im{y_prime} = 0 for real y, so just keep real.

	# Calculate the polynomials in x necessary for transforming back to the Chebyshev domain
	numers = deque([poly([-1])]) # just -1 to start, at order 1
	denom = poly([1, 0, -1]) # 1 - x^2
	for nu in range(2, order + 1): # initialization takes care of order 1, so iterate from order 2
		q = 0
		for mu in range(1, nu): # Terms come from the previous derivative, so there are nu - 1 of them here.
			p = numers.popleft() # c = nu - mu/2
			numers.append(denom * p.deriv() + (nu - mu/2 - 1) * poly([0, 2]) * p - q)
			q = p
		numers.append(-q)
	
	# Calculate x derivative as a sum of x polynomials * theta-domain derivatives
	dy_n = np.zeros(y_n.shape) # The middle of dy will get filled with a derivative expression in terms of y_primes
	denom_x = denom(x_n[1:-1]) # only calculate this once; leave off +/-1, because they need to be treated specially anyway
	for term,(numer,y_prime) in enumerate(zip(numers, y_primes), 1): # iterating from lower derivatives to higher
		c = order - term/2 # c starts at nu - 1/2 and then loses 1/2 for each subsequent term
		dy_n[middle] += (numer(x_n[1:-1])/(denom_x**c))[s] * y_prime

	# Calculate the endpoints
	if order <= 4 and calc_endpoints:
		C, D = {1: [(-1,), 1], 2: [(1, 1), 3], 3: [(-4, -5, -1), 15], 4: [(36, 49, 14, 1), 105]}[order] # Constants from the math. See the notebook in the warning.
		LH = 0 # L'Hôpital numerator terms
		for i,C_i in enumerate(C, 1): # i starts at 1
			LH += 2 * C_i * (-1)**i * np.power(k, 2*i)
			if dct_type == 1: LH[-1] -= C_i * (-1)**i * np.power(N, 2*i) # because Nth element is outside the 2\sum in the DCT-I
		dy_n[first] = np.sum(LH[s] * Y_k, axis=axis)/ (D*M)
		dy_n[last] = np.sum((LH * np.power(-1, k))[s] * Y_k, axis=axis) / ((-1)**order * D*M)
	else: # For higher derivatives, leave the endpoints uncalculated, but direct the user to my analysis of this problem.
		if calc_endpoints: warn("""endpoints set to NaN, only calculated for 4th derivatives and below. For help with higher derivatives,
			see https://github.com/pavelkomarov/spectral-derivatives/blob/main/notebooks/chebyshev_domain_endpoints.ipynb""")
		dy_n[first] = np.nan
		dy_n[last] = np.nan

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
	#No worrying about conversion back from a variable transformation. No special treatment of domain boundaries.
	if len(t_n.shape) > 1 or t_n.shape[0] != y_n.shape[axis]:
		raise ValueError("t_n should be 1D and have the same length as y_n along the axis of differentiation")
	if not np.all(np.diff(t_n) > 0):
		raise ValueError("The domain, t_n, should be ordered low-to-high, [a, ... b). Try sampling with `np.arange(0, M)/M * (b - a) + a`")

	M = y_n.shape[axis]
	# if M has an even length, then we make k = [0, 1, ... M/2 - 1, 0 or M/2, -M/2 + 1, ... -1]
	# if M has odd length, k = [0, 1, ... floor(M/2), -floor(M/2), ... -1]
	k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))
	if M % 2 == 0 and order % 2 == 1: k[M//2] = 0 # odd derivatives get the Nyquist element zeroed out, if there is one

	s = [np.newaxis for dim in y_n.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y_k = np.fft.fft(y_n, axis=axis)
	if filter: Y_k *= filter(k)[s]
	with catch_warnings(): simplefilter("ignore", category=RuntimeWarning); Y_nu = (1j * k[s])**order * Y_k # if nu < 0, we're dividing by 0
	if order < 0: Y_nu[np.where(k==0)] = 0; warn("+c information lost in antiderivative") # Get rid of NaNs. Enables taking the antiderivative.
	dy_n = np.fft.ifft(Y_nu, axis=axis).real if not np.iscomplexobj(y_n) else np.fft.ifft(Y_nu, axis=axis)

	# The above is agnostic to where the data came from, pretends it came from the domain [0, 2pi), but the data may actually
	scale = (t_n[M-1] + t_n[1] - 2*t_n[0])/(2*np.pi) # be smooshed from some other domain. So scale the derivative by the
	return dy_n/scale**order 						# relative size of the t and theta intervals.
