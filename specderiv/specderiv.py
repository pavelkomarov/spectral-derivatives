import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.fft import dct, dst
from collections import deque
from warnings import warn


def cheb_deriv(t_n: np.ndarray, y_n: np.ndarray, nu: int, axis: int=0):
	"""Evaluate derivatives with Chebyshev polynomials via discrete cosine and sine transforms. Caveats:

	- Taking the 1st derivative twice with a discrete method like this is not exactly the same as taking the second derivative.
	- For derivatives over the 4th, this method presently returns :code:`NaN` at the edges of the domain. Be cautious if passing
	  the result to another function.

	Args:
		t_n (np.ndarray): Where the function :math:`y` is sampled in the dimension of differentation. If you're using canonical
			Chebyshev points, this will be :code:`x_n = np.cos(np.arange(N+1) * np.pi / N)` (:math:`x \\in [1, -1]`). If you're
			sampling on a domain from :math:`a` to :math:`b`, this needs to be :code:`t_n = np.cos(np.arange(N+1) * np.pi / N) *
			(b - a)/2 + (b + a)/2`. Note the order is high-to-low in the :math:`x` or :math:`t` domain, but low-to-high in
			:math:`n`. Also note both endpoints are *inclusive*.
		y_n (np.ndarray): Data to transform, representing a function sampled at the cosine-spaced points in each dimension.
		nu (int): The order of derivative to take.
		axis (int, optional): The dimension along which to take the derivative. Defaults to the first dimension (axis=0).
 
	Returns:
		np.ndarray: :code:`dy`, data representing the :math:`\\nu^{th}` derivative of the function, sampled at points :math:`x_n`
	"""
	N = y_n.shape[axis] - 1; M = 2*N # We only have to care about the number of points in the dimension we're differentiating

	if not np.all(np.diff(t_n) < 0):
		raise ValueError("The domain, t_n, should be ordered high-to-low, [b, ... a]. Try sampling with `np.cos(np.arange(N+1) * np.pi / N) * (b - a)/2 + (b + a)/2`")
	scale = (t_n[0] - t_n[N])/2; offset = (t_n[0] + t_n[N])/2 # Trying to be helpful, because sampling is tricky to get right
	if not np.allclose(t_n, np.cos(np.arange(N+1) * np.pi / N) * scale + offset):
		raise ValueError("Your function is not sampled at cosine-spaced points! Try sampling with `np.cos(np.arange(N+1) * np.pi / N) * (b - a)/2 + (b + a)/2`")

	first = [slice(None) for dim in y_n.shape]; first[axis] = 0; first = tuple(first) # for accessing different parts of data
	last = [slice(None) for dim in y_n.shape]; last[axis] = N; last = tuple(last)
	middle = [slice(None) for dim in y_n.shape]; middle[axis] = slice(1, -1); middle = tuple(middle)
	s = [np.newaxis for dim in y_n.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y_k = dct(y_n, 1, axis=axis) # Transform to frequency domain using the 1st definition of the discrete cosine transform
	k = np.arange(1, N) # [1, ... N-1], wavenumber iterator/indices

	y_primes = [] # Store all derivatives in theta up to the nu^th, because we need them all for reconstruction.
	for order in range(1, nu+1):
		if order % 2: # odd derivative
			Y_order = (1j * k[s])**order * Y_k[middle] # Y_prime[k=0 and N] = 0 and so are not needed for the DST
			y_primes.append(dst(1j * Y_order, 1, axis=axis).real / M) # d/dtheta y = the inverse transform of DST-1 
				# = 1/M * DST-1. Extra j for equivalence with IFFT. Im{y_prime} = 0 for real y, so just keep real.
		else: # even derivative
			Y_order = (1j * np.arange(0, N+1)[s])**order * Y_k # Include terms for wavenumbers 0 and N, becase the DCT uses them
			y_primes.append(dct(Y_order, 1, axis=axis)[middle].real / M) # the inverse transform of DCT-1 is 1/M * DCT-1.
				# Slice off ends. Im{y_prime} = 0 for real y, so just keep real.

	# Calculate the polynomials in x necessary for transforming back to the Chebyshev domain
	numers = deque([poly([-1])]) # just -1 to start, at order 1
	denom = poly([1, 0, -1]) # 1 - x^2
	for order in range(2, nu + 1): #
		q = 0
		for term in range(1, order): # Terms come from the previous derivative, so there are order-1 of them here.
			p = numers.popleft() # c = order - term/2
			numers.append(denom * p.deriv() + (order - term/2 - 1) * poly([0, 2]) * p - q)
			q = p
		numers.append(-q)
	
	# Calculate x derivative as a sum of x polynomials * theta-domain derivatives
	dy_n = np.zeros(y_n.shape) # The middle of dy will get filled with a derivative expression in terms of y_primes
	x_n = np.cos(np.pi * np.arange(1, N) / N) # leave off +/-1, because they need to be treated specially anyway
	denom_x = denom(x_n) # only calculate this once
	for term,(numer,y_prime) in enumerate(zip(numers, y_primes), 1): # iterating from lower derivatives to higher
		c = nu - term/2 # c starts at nu - 1/2 and then loses 1/2 for each subsequent term
		dy_n[middle] += (numer(x_n)/(denom_x**c))[s] * y_prime

	if nu == 1: # Fill in the endpoints. Unfortunately this takes special formulas for each nu.
		dy_n[first] = np.sum((k**2)[s] * Y_k[middle], axis=axis)/N + (N/2) * Y_k[last]
		dy_n[last] = -np.sum((k**2 * np.power(-1, k))[s] * Y_k[middle], axis=axis)/N - (N/2)*(-1)**N * Y_k[last]
	elif nu == 2: # And they're not short formulas either :(
		dy_n[first] = np.sum((k**4 - k**2)[s] * Y_k[middle], axis=axis)/(3*N) + (N/6)*(N**2 - 1) * Y_k[last]
		dy_n[last] = np.sum(((k**4 - k**2)*np.power(-1, k))[s] * Y_k[middle], axis=axis)/(3*N) + (N/6)*(N**2 - 1)*(-1)**N * Y_k[last] 
	elif nu == 3:
		dy_n[first] = np.sum((k**6 - 5*k**4 + 4*k**2)[s] * Y_k[middle], axis=axis)/(15*N) + N*((N**4)/30 - (N**2)/6 + 2/15)*Y_k[last]
		dy_n[last] = -np.sum(((k**6 - 5*k**4 + 4*k**2)*np.power(-1, k))[s] * Y_k[middle], axis=axis)/(15*N) - N*((N**4)/30 - (N**2)/6 + 2/15)*(-1)**N * Y_k[last]
	elif nu == 4:
		dy_n[first] = np.sum((k**8 - 14*k**6 + 49*k**4 - 36*k**2)[s] * Y_k[middle], axis=axis)/(105*N) + N*(N**6 - 14*N**4 + 49*N**2 - 36)/210 * Y_k[last]
		dy_n[last] = np.sum(((k**8 - 14*k**6 + 49*k**4 - 36*k**2)*np.power(-1, k))[s] * Y_k[middle], axis=axis)/(105*N) + (N*(N**6 - 14*N**4 + 49*N**2 - 36)*(-1)**N)/210 * Y_k[last]
	else: # For higher derivatives, leave the endpoints uncalculated
		warn("endpoints set to NaN, only calculated for 4th derivatives and below")
		dy_n[first] = np.nan
		dy_n[last] = np.nan

	# The above is agnostic to where the data came from, pretends it came from the domain [-1, 1], but the data is
	return dy_n/scale**nu # actually smooshed from some other domain. So scale the derivative appropriately.


def fourier_deriv(t_n: np.ndarray, y_n: np.ndarray, nu: int, axis: int=0):
	"""For use with periodic functions.
 
	Args:
		t_n (np.ndarray): Where the function :math:`y` is sampled in the dimension of differentiation. If you're using canonical
			Fourier points, this will be :code:`th_n = np.arange(M) * 2*np.pi / M` (:math:`\\theta \\in [0, 2\\pi)`). If you're
			sampling on a domain from :math:`a` to :math:`b`, this needs to be :code:`t_n = np.arange(0, M)/M * (b - a) + a`.
			Note the lower, left bound is *inclusive* and the upper, right bound is *exclusive*.
		y_n (np.ndarray): Data to transform, representing a period of a periodic function, sampled at equispaced points in
			each dimension.
		nu (int): The order of derivative to take.
		axis (int, optional): The dimension along which to take the derivative. Defaults to the first dimension (axis=0).

	Returns:
		np.ndarray: :code:`dy`, data representing the :math:`\\nu^{th}` derivative of the function, sampled at points :math:`x_n`
	"""
	#No worrying about conversion back from a variable transformation. No special treatment of domain boundaries.
	if not np.all(np.diff(t_n) > 0):
		raise ValueError("The domain, t_n, should be ordered low-to-high, [a, ... b). Try sampling with `np.arange(0, M)/M * (b - a) + a`")

	M = y_n.shape[axis]
	if M % 2 == 0: # if M has an even length, then we make k = [0, 1, ... M/2 - 1, 0 or M/2, -M/2 + 1, ... -1]
		k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))
		if nu % 2 == 1: # odd derivatives get the M/2th element zeroed out
			k[M//2] = 0
	else: # M has odd length, so k = [0, 1, ... floor(M/2), -floor(M/2), ... -1]
		k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))

	s = [np.newaxis for dim in y_n.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y_k = np.fft.fft(y_n, axis=axis)
	Y_nu = (1j * k[s])**nu * Y_k
	dy_n = np.fft.ifft(Y_nu, axis=axis).real if not np.iscomplexobj(y_n) else np.fft.ifft(Y_nu, axis=axis)

	# The above is agnostic to where the data came from, pretends it came from the domain [0, 2pi), but the data is
	scale = (t_n[M-1] + t_n[1] - 2*t_n[0])/(2*np.pi) # actually smooshed from some other domain. So scale the derivative appropriately.
	return dy_n/scale**nu 
