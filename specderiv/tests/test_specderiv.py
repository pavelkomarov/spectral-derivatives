# run with python(3) -m pytest
# CI runs this with `coverage` and uploads results to coveralls, badge displayed in the readme

import pytest
import numpy as np
from ..specderiv import cheb_deriv, fourier_deriv

# for use with Chebyshev
N = 20
x_n = np.cos(np.arange(N+1) * np.pi / N) # length N+1, in keeping with the usage of N in Trefethen.

# for use with Fourier
M = 20					# It's important this be an *open* periodic domain, or we get artefacting
th_n = np.arange(0, M) * 2*np.pi / M # e.g. th_n = np.linspace(0, M, M)*2*np.pi/M is no good

@pytest.mark.filterwarnings('ignore::UserWarning') # Not worrying about warnings in this test
def test_cheb_deriv_accurate_to_6th():
	"""A test that the MSE of derivatives of a non-periodic function vs truth is suitably low up to some
	high power. Implicitly tests the middle polynomial-finding code, the endpoints-finding code, and even
	derivatives as well as odd derivatives.
	"""
	y_n = np.exp(x_n) * np.sin(5*x_n)
	analytic_truth = [5*np.exp(x_n) * np.cos(5*x_n) + np.exp(x_n) * np.sin(5*x_n),	# 1st
						2*np.exp(x_n) * (5*np.cos(5*x_n) - 12*np.sin(5*x_n)),		# 2nd
						-2*np.exp(x_n) * (37*np.sin(5*x_n) + 55*np.cos(5*x_n)),		# 3rd
						4*np.exp(x_n) * (119*np.sin(5*x_n) - 120*np.cos(5*x_n)),	# 4th
						4*np.exp(x_n) * (719*np.sin(5*x_n) + 475*np.cos(5*x_n)),	# 5th
						8*np.exp(x_n) * (2035*np.cos(5*x_n) - 828*np.sin(5*x_n))]	# 6th
	# Things get less accurate for higher derivatives, so check < 10^f(nu)
	L2_powers = [-19, -14, -10, -6, -3, 0]
	L1_powers = [-9, -6, -4, -2, -1, 1]

	for nu in range(1,7):
		computed = cheb_deriv(y_n, nu)
		assert np.nanmean((analytic_truth[nu-1] - computed)**2) < 10**L2_powers[nu-1]
		assert np.nanmax(np.abs(analytic_truth[nu-1] - computed)) < 10**L1_powers[nu-1]

def test_cheb_endpoints():
	"""A test of the endpoints code, specifically. Endpoints should be found accurately up to the
	4th derivative, NaN at derivatives beyond the 4th, and a warning should be thrown about the presence
	of NaNs in the answer.
	"""
	y_n = 3*x_n**6 - 2*x_n**4
	analytic_truth = [18*x_n**5 - 8*x_n**3,			# 1st
						90*x_n**4 - 24*x_n**2,		# 2nd
						360*x_n**3 - 48*x_n,		# 3rd
						1080*x_n**2 - 48,			# 4th
						2160*x_n,					# 5th
						2160*np.ones(x_n.shape)]	# 6th

	for nu in range(1, 7):
		if nu <= 4:
			computed = cheb_deriv(y_n, nu)
			assert np.abs(computed[0] - analytic_truth[nu-1][0]) < 1e-7
			assert np.abs(computed[-1] - analytic_truth[nu-1][-1]) < 1e-7
		else: # nu > 4
			with pytest.warns(UserWarning): # assure the warning is thrown
				computed = cheb_deriv(y_n, nu)
			assert np.isnan(computed[0]) # the endpoints are NaN
			assert np.isnan(computed[-1])
			assert np.all(~np.isnan(computed[1:-1])) # the middle isn't NaN
		assert np.nanmean(analytic_truth[nu-1] - computed)**2 < 1e-7 # check middle too for good measure

def test_fourier_deriv_accurate_to_3rd():
	"""A test for derivatives of a periodic function sampled at equispaced points
	"""
	y_n = np.cos(th_n) + 2*np.sin(3*th_n)
	analytic_truth = [-np.sin(th_n) + 6*np.cos(3*th_n),		# 1st
						-np.cos(th_n) - 18*np.sin(3*th_n),	# 2nd
						np.sin(th_n) - 54*np.cos(3*th_n)]	# 3rd

	for nu in range(1,4): # Things get less accurate for higher derivatives, so check < 10^f(nu)
		computed = fourier_deriv(y_n, nu)
		assert np.nanmean((analytic_truth[nu-1] - computed)**2) < 1e-25
		assert np.nanmax(np.abs(analytic_truth[nu-1] - computed)) < 1e-12

def test_cheb_multidimensional():
	"""A test for multidimensional derivatives in the aperiodic case
	"""
	X1_n, X2_n = np.meshgrid(x_n, x_n) # a 100 x 100 grid
	y_n = X1_n**2 * np.sin(3/2 * np.pi * X2_n)
	
	# d^2 / dx_1 dx_2
	analytic_truth = 3 * X1_n * np.pi * np.cos(3/2 * np.pi * X2_n)
	computed = cheb_deriv(cheb_deriv(y_n, 1, axis=0), 1, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-18
	assert np.max(np.abs(analytic_truth - computed)) < 1e-9
	
	# Laplacian
	analytic_truth = 2 * np.sin(3/2 * np.pi * X2_n) - 9/4 * np.pi**2 * X1_n**2 * np.sin(3/2 * np.pi * X2_n)
	computed = cheb_deriv(y_n, 2, axis=0) + cheb_deriv(y_n, 2, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-16
	assert np.max(np.abs(analytic_truth - computed)) < 1e-6

def test_fourier_multidimensional():
	"""A test for multidimensional derivatives in the periodic case
	"""
	T1_n, T2_n = np.meshgrid(th_n, th_n) # a 100 x 100 grid
	y_n = np.sin(2*T1_n) * np.cos(T2_n)

	#d^2 / d theta_1 d theta_2
	analytic_truth = -2 * np.cos(2 * T1_n) * np.sin(T2_n)
	computed = fourier_deriv(fourier_deriv(y_n, 1, axis=0), 1, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-25
	assert np.max(np.abs(analytic_truth - computed)) < 1e-14

	# Laplacian
	analytic_truth = -5 * np.sin(2 * T1_n) * np.cos(T2_n)
	computed = fourier_deriv(y_n, 2, axis=0) + fourier_deriv(y_n, 2, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-25
	assert np.max(np.abs(analytic_truth - computed)) < 1e-13
