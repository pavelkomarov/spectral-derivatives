import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.fftpack import dct, dst
from collections import deque
from matplotlib import pyplot

def cheb_deriv(y, nu, axis=0):
	"""Evaluate derivatives with Chebyshev polynomials via discrete cosine and sine transforms
	:param y: Data to transform, representing a function at Chebyshev points in each dimension x_n = cos(pi * n /N)
	:param nu: The order of derivative to take
	:param axis: The dimension of along which to take the derivative
	:return dy: data representing the nu^th derivative of the function, sampled at points x_n
	"""
	N = y.shape[axis] - 1; M = 2*N # We only have to care about the number of points in the dimension we're differentiating
	if N == 0: return 0 # because if the function is a constant, the derivative is 0

	first = [slice(None) for dim in y.shape]; first[axis] = 0; first = tuple(first) # for accessing different parts of data
	last = [slice(None) for dim in y.shape]; last[axis] = N; last = tuple(last)
	middle = [slice(None) for dim in y.shape]; middle[axis] = slice(1, -1); middle = tuple(middle)
	s = [np.newaxis for dim in y.shape]; s[axis] = slice(None); s = tuple(s) # for elevating vectors to have same dimension as data

	Y = dct(y, 1, axis=axis) # Transform to frequency domain using the 1st definition of the discrete cosine transform
	k = np.arange(1, N) # [1, ... N-1], wavenumber iterator/indices

	y_primes = [] # Store all derivatives in theta up to the nu^th, because we need them all for reconstruction
	for order in range(1, nu+1):
		if order % 2: # odd derivative
			Y_prime = (1j * k[s])**order * Y[middle] # Y_prime[k=0 and N] = 0 and so are not needed for the DST
			y_primes.append(dst(1j * Y_prime, 1, axis=axis).real / M) # d/dtheta y = the inverse transform of DST-1 
				# = 1/M * DST-1. Extra j for equivalence with IFFT. Im{y_prime} = 0 for real y, so just keep real.
		else: # even derivative
			Y_prime = (1j * np.arange(0, N+1)[s])**order * Y # Include terms for wavenumbers 0 and N, becase the DCT uses them
			y_primes.append(dct(Y_prime, 1, axis=axis)[middle].real / M) # the inverse transform of DCT-1 is 1/M * DCT-1.
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
		dy[first] = np.nan
		dy[last] = np.nan

	return dy

N = 20
x_n = np.cos(np.linspace(0, N, N+1) * np.pi / N)

example = 1

# 1D example
if example == 1:
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

	y = np.exp(x_n) * np.sin(5*x_n)
	dy = cheb_deriv(y, 1)
	d2y = cheb_deriv(y, 2)
	d3y = cheb_deriv(y, 3)
	d4y = cheb_deriv(y, 4)
	#d5y = cheb_deriv(y, 5)
	#d6y = cheb_deriv(y, 6)

	pyplot.plot(x_n, y, 'k+')
	pyplot.plot(x_n, dy, 'b+')
	pyplot.plot(x_n, d2y, 'r+')
	pyplot.plot(x_n, d3y, 'g+')
	pyplot.plot(x_n, d4y, 'm+')
	#pyplot.plot(x_n, d5y, 'c+')
	#pyplot.plot(x_n, d6y, 'y+')

	pyplot.show()

if example == 2:
	# 2D example
	x = np.linspace(-1, 1, 100)
	y = np.linspace(-1, 1, 100)
	X, Y = np.meshgrid(x, y)

	F = X**2 * np.sin(3/2*np.pi*Y)
	dxdyF = 3*X*np.pi*np.cos(3/2*np.pi*Y) # d^2 / dx dy
	Laplacian = 2*np.sin(3/2*np.pi*Y) - 9/4*np.pi**2 * X**2 * np.sin(3/2*np.pi*Y)

	fig = pyplot.figure(figsize=(12, 5))
	ax1 = fig.add_subplot(1, 3, 1, projection='3d')
	ax1.plot_surface(X, Y, F, cmap='viridis', alpha=0.5)
	ax1.set_title('original function')
	ax1.set_xlabel('x')
	ax1.set_ylabel('y')
	ax2 = fig.add_subplot(1, 3, 2, projection='3d')
	ax2.plot_surface(X, Y, dxdyF, cmap='viridis', alpha=0.5)
	ax2.set_title(r'$\frac{d^2}{dxdy}$')
	ax2.set_xlabel('x')
	ax2.set_ylabel('y')
	ax2.set_zlim((-8,8))
	ax3 = fig.add_subplot(1, 3, 3, projection='3d')
	ax3.plot_surface(X, Y, Laplacian, cmap='viridis', alpha=0.5)
	ax3.set_title(r'$\frac{d^2}{dx^2} + \frac{d^2}{dy^2}$')
	ax3.set_xlabel('x')
	ax3.set_ylabel('y')
	ax3.set_zlim((-20, 20))

	X_n, Y_n = np.meshgrid(x_n, x_n) # same shapes as X and Y, but cosine spacing
	F_n = X_n**2 * np.sin(3/2*np.pi*Y_n) # F sampled at the cosine-spaced points

	#dF_n = np.zeros((N+1, N+1))
	# for i in range(N+1): # iterate first dimension, taking derivatives along second dimension
	# 	dF_n[i] = cheb_deriv(F_n[i], 1)
	# for j in range(N+1): # iterate second dimension, taking derivatives along the first
	# 	dF_n[:,j] = cheb_deriv(dF_n[:,j], 1)
	dF_n = cheb_deriv(cheb_deriv(F_n, 1, axis=0), 1, axis=1) # One-lineable!

	#Laplacian_n = np.zeros((N+1, N+1))
	# for i in range(N+1): # iterate first dimension, taking derivatives along second dimension
	# 	Laplacian_n[i] += cheb_deriv(F_n[i], 2)
	# for j in range(N+1): # iterate second dimension, taking derivatives along the first 
	# 	Laplacian_n[:,j] += cheb_deriv(F_n[:,j], 2)
	Laplacian_n = cheb_deriv(F_n, 2, axis=0) + cheb_deriv(F_n, 2, axis=1) # One-lineable!

	ax1.plot_wireframe(X_n, Y_n, F_n)
	ax2.plot_wireframe(X_n, Y_n, dF_n)
	ax3.plot_wireframe(X_n, Y_n, Laplacian_n)

	pyplot.tight_layout()
	pyplot.show()
