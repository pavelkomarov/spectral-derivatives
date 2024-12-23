import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.fftpack import dct, dst
from collections import deque
from matplotlib import pyplot

def cheb_deriv(y, nu, axis=-1):
	"""Evaluate derivatives with Chebyshev polynomials via discrete cosine and sine transforms
	:param y: Data to transform, representing a function at Chebyshev points in each dimension x_n = cos(pi * n /N)
	:param nu: The order of derivative to take
	:param axis: The dimension of along which to take the derivative
	:return: A vector represending the nu^th derivative of the function, sampled at points x_n
	"""
	#if not 0 < nu < 5: raise NotImplementedError("This function only supports derivatives between order 1 and 4.")
	N = np.size(y) - 1; M = 2*N
	if N == 0: return 0 # because if the function is a constant, the derivative is 0

	Y = dct(y, 1, axis=axis) # Transform to frequency domain using the 1st definition of the discrete cosine transform
	k = np.arange(0, N+1) # [0, ... N], wavenumber iterator/indices

	y_primes = [] # Store all derivatives in theta up to the nu^th, because we need them all for reconstruction
	for order in range(1, nu+1):
		if order % 2: # odd derivative
			Y_prime = (1j * k[1:-1])**order * Y[1:-1] # Y_prime[k=0 and N] = 0 and so are not needed for the DST
			y_primes.append(dst(1j * Y_prime, 1, axis=axis).real / M) # d/dtheta y = the inverse transform of DST-1,
			# which is 1/M * DST-1. Extra j for equivalence with IFFT. Im{y_prime} = 0 for real y.
		else: # even derivative
			Y_prime = (1j * k)**order * Y # Include terms for wavenumbers 0 and N, becase the DCT uses them
			y_primes.append(dct(Y_prime, 1, axis=axis)[1:-1].real / M) # the inverse transform of DCT-1 is 1/M * DCT-1.
			# Slice off ends. Im{y_prime} = 0 for real y.

	# n = k[1:-1] = [1, ... N-1]. n iterates space, as opposed to k, which iterates wavenumbers, but reuse a vector
	x = np.cos(np.pi * k[1:-1] / N) # leave off +/-1, because they need to be treated specially anyway
	dy = np.empty(y.shape) # The middle of dy will get filled with a derivative expression in terms of y_primes

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
	for term,(numer,y_prime) in enumerate(zip(numers, y_primes), 1): # iterating from lower derivatives to higher
		c = nu - term/2 # c starts at nu - 1/2 and then loses 1/2 for each subsequent term
		dy[1:-1] += (numer(x)/(denom(x)**c)) * y_prime

	if nu == 1: # Fill in the endpoints. Unfortunately this takes special formulas for each nu.
		dy[0] = np.sum(k[1:-1]**2 * Y[1:-1])/N + (N/2) * Y[N]
		dy[N] = -np.sum(k[1:-1]**2 * Y[1:-1] * np.power(-1, k[1:-1]))/N - (N/2) * Y[N] * (-1)**N
	elif nu == 2: # And they're not short formulas either :(
		dy[0] = np.sum((k[1:-1]**4 - k[1:-1]**2) * Y[1:-1])/(3*N) + (N/6)*(N**2 - 1) * Y[N]
		dy[N] = np.sum((k[1:-1]**4 - k[1:-1]**2) * Y[1:-1] * np.power(-1, k[1:-1]))/(3*N) + (N/6)*(N**2 - 1) * Y[N] * (-1)**N
	elif nu == 3: # I derived this multiple times, but it doesn't seem to work. Not sure where the error is.
		dy[0] = np.sum(((k[1:-1]**6)/15 - (k[1:-1]**4)/3 + 4*(k[1:-1]**2)/15) * Y[1:-1])/N + N*((N**4)/30 - (N**2)/6 + 2/15)*Y[N]
		dy[N] = -np.sum(((k[1:-1]**6)/15 - (k[1:-1]**4)/3 + 4*(k[1:-1]**2)/15) * Y[1:-1] * np.power(-1, k[1:-1]))/N - N*((N**4)/30 - (N**2)/6 + 2/15)*Y[N]*(-1)**N
	else: # For higher derivatives, leave the endpoints uncalculated
		dy[0] = np.nan
		dy[N] = np.nan

	return dy

# Try it out on the function e^x sin(5x)
x = np.linspace(-1, 1, 100)

f = np.exp(x) * np.sin(5*x)
df = 5*np.exp(x) * np.cos(5*x) + f
d2f = 2*np.exp(x) * (5*np.cos(5*x) - 12*np.sin(5*x))
d3f = -2*np.exp(x) * (37*np.sin(5*x) + 55*np.cos(5*x))
d4f = 4*np.exp(x) * (119*np.sin(5*x) - 120*np.cos(5*x))
#d5f = 4*np.exp(x) * (719*np.sin(5*x) + 475*np.cos(5*x))
#d6f = 8*np.exp(x) * (2035*np.cos(5*x) - 828*np.sin(5*x))

pyplot.plot(x, f, 'k')
pyplot.plot(x, df, 'b')
pyplot.plot(x, d2f, 'r')
pyplot.plot(x, d3f, 'g')
pyplot.plot(x, d4f, 'm')
#pyplot.plot(x, d5f, 'c')
#pyplot.plot(x, d6f, 'y')

N = 20
x_c = np.cos(np.linspace(0, N, N+1) * np.pi / N)
y = np.exp(x_c) * np.sin(5*x_c)
dy = cheb_deriv(y, 1)
d2y = cheb_deriv(y, 2)
d3y = cheb_deriv(y, 3)
d4y = cheb_deriv(y, 4)
#d5y = cheb_deriv(y, 5)
#d6y = cheb_deriv(y, 6)

pyplot.plot(x_c, y, 'k+')
pyplot.plot(x_c, dy, 'b+')
pyplot.plot(x_c, d2y, 'r+')
pyplot.plot(x_c, d3y, 'g+')
pyplot.plot(x_c, d4y, 'm+')
#pyplot.plot(x_c, d5y, 'c+')
#pyplot.plot(x_c, d6y, 'y+')

pyplot.show()
