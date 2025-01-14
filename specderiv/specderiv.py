import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.fft import dct, dst
from collections import deque
from warnings import warn


def cheb_deriv(y, nu, axis=0):
	"""Evaluate derivatives with Chebyshev polynomials via discrete cosine and sine transforms. Caveats:

	- Taking the 1st derivative twice with a discrete method like this is not exactly the same as taking the second derivative.
	- For derivatives over the 4th, this method presently returns :code:`NaN` at the edges of the domain. Be cautious if passing
	  the result to another function.

	:param y: Data to transform, representing a function at Chebyshev points in each dimension :math:`x_n = cos(\\frac{\\pi n}{N}), n \\in [0, N-1]`
	:param nu: The order of derivative to take
	:param axis: The dimension along which to take the derivative, defaults to first dimension
	:return: :code:`dy`, data representing the :math:`\\nu^{th}` derivative of the function, sampled at points :math:`x_n`
	"""
	N = y.shape[axis] - 1; M = 2*N # We only have to care about the number of points in the dimension we're differentiating
	if N == 0: return 0 # because if the function is a constant, the derivative is 0

	first = [slice(None) for dim in y.shape]; first[axis] = 0; first = tuple(first) # for accessing different parts of data
	last = [slice(None) for dim in y.shape]; last[axis] = N; last = tuple(last)
	middle = [slice(None) for dim in y.shape]; middle[axis] = slice(1, -1); middle = tuple(middle)
	s = [np.newaxis for dim in y.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y = dct(y, 1, axis=axis) # Transform to frequency domain using the 1st definition of the discrete cosine transform
	k = np.arange(1, N) # [1, ... N-1], wavenumber iterator/indices

	y_primes = [] # Store all derivatives in theta up to the nu^th, because we need them all for reconstruction.
	for order in range(1, nu+1):
		if order % 2: # odd derivative
			Y_order = (1j * k[s])**order * Y[middle] # Y_prime[k=0 and N] = 0 and so are not needed for the DST
			y_primes.append(dst(1j * Y_order, 1, axis=axis).real / M) # d/dtheta y = the inverse transform of DST-1 
				# = 1/M * DST-1. Extra j for equivalence with IFFT. Im{y_prime} = 0 for real y, so just keep real.
		else: # even derivative
			Y_order = (1j * np.arange(0, N+1)[s])**order * Y # Include terms for wavenumbers 0 and N, becase the DCT uses them
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
	
	#Calculate x derivative as a sum of x polynomials * theta-domain derivatives
	dy = np.zeros(y.shape) # The middle of dy will get filled with a derivative expression in terms of y_primes
	x = np.cos(np.pi * np.arange(1, N) / N) # leave off +/-1, because they need to be treated specially anyway
	denom_x = denom(x) # only calculate this once
	for term,(numer,y_prime) in enumerate(zip(numers, y_primes), 1): # iterating from lower derivatives to higher
		c = nu - term/2 # c starts at nu - 1/2 and then loses 1/2 for each subsequent term
		dy[middle] += (numer(x)/(denom_x**c))[s] * y_prime

	if nu == 1: # Fill in the endpoints. Unfortunately this takes special formulas for each nu.
		dy[first] = np.sum((k**2)[s] * Y[middle], axis=axis)/N + (N/2) * Y[last]
		dy[last] = -np.sum((k**2 * np.power(-1, k))[s] * Y[middle], axis=axis)/N - (N/2)*(-1)**N * Y[last]
	elif nu == 2: # And they're not short formulas either :(
		dy[first] = np.sum((k**4 - k**2)[s] * Y[middle], axis=axis)/(3*N) + (N/6)*(N**2 - 1) * Y[last]
		dy[last] = np.sum(((k**4 - k**2)*np.power(-1, k))[s] * Y[middle], axis=axis)/(3*N) + (N/6)*(N**2 - 1)*(-1)**N * Y[last] 
	elif nu == 3:
		dy[first] = np.sum((k**6 - 5*k**4 + 4*k**2)[s] * Y[middle], axis=axis)/(15*N) + N*((N**4)/30 - (N**2)/6 + 2/15)*Y[last]
		dy[last] = -np.sum(((k**6 - 5*k**4 + 4*k**2)*np.power(-1, k))[s] * Y[middle], axis=axis)/(15*N) - N*((N**4)/30 - (N**2)/6 + 2/15)*(-1)**N * Y[last]
	elif nu == 4:
		dy[first] = np.sum((k**8 - 14*k**6 + 49*k**4 - 36*k**2)[s] * Y[middle], axis=axis)/(105*N) + N*(N**6 - 14*N**4 + 49*N**2 - 36)/210 * Y[last]
		dy[last] = np.sum(((k**8 - 14*k**6 + 49*k**4 - 36*k**2)*np.power(-1, k))[s] * Y[middle], axis=axis)/(105*N) + (N*(N**6 - 14*N**4 + 49*N**2 - 36)*(-1)**N)/210 * Y[last]
	else: # For higher derivatives, leave the endpoints uncalculated
		warn("endpoints set to NaN, only calculated for 4th derivatives and below")
		dy[first] = np.nan
		dy[last] = np.nan

	return dy


def fourier_deriv(y, nu, axis=0):
	"""For use with periodic functions.

	:param y: Data to transform, representing a function at equispaced points :math:`\\theta_n \\in [0, 2\\pi)`
	:param nu: The order of derivative to take
	:param axis: The dimension along which to take the derivative, defaults to first dimension
	:return: :code:`dy`, data representing the :math:`\\nu^{th}` derivative of the function, sampled at points :math:`\\theta_n`
	"""
	#No worrying about conversion back from a variable transformation. No special treatment of domain boundaries.
	M = y.shape[axis]
	if M % 2 == 0: # if M has an even length, then we make k = [0, 1, ... M/2 - 1, 0 or M/2, -M/2 + 1, ... -1]
		k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))
		if nu % 2 == 1: # odd derivatives get the M/2th element zeroed out
			k[M//2] = 0
	else: # M has odd length, so k = [0, 1, ... floor(M/2), -floor(M/2), ... -1]
		k = np.concatenate((np.arange(M//2 + 1), np.arange(-M//2 + 1, 0)))

	s = [np.newaxis for dim in y.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y = np.fft.fft(y, axis=axis)
	Y_nu = (1j * k[s])**nu * Y
	return np.fft.ifft(Y_nu, axis=axis).real if not np.iscomplexobj(y) else np.fft.ifft(Y_nu, axis=axis)
