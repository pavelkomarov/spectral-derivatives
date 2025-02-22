# run with python(3) -m pytest -s
# CI runs this with `coverage` and uploads results to coveralls, badge displayed in the readme

import pytest
import numpy as np
from ..specderiv import cheb_deriv, fourier_deriv

# for use with Chebyshev
N = 20
x_n = np.cos(np.arange(N+1) * np.pi / N) # length N+1, in keeping with the usage of N in Trefethen.
x_n2 = np.concatenate(([1], np.cos(np.pi/(N+1) * (np.arange(N+1) + 0.5)), [-1])) # use half-indices for DCT-II

# for use with Fourier
M = 20					# It's important this be an *open* periodic domain, or we get artefacting
th_n = np.arange(0, M) * 2*np.pi / M # e.g. th_n = np.linspace(0, M, M)*2*np.pi/M is no good

@pytest.mark.filterwarnings('ignore::UserWarning') # Not worrying about warnings in this test
@pytest.mark.parametrize("dct_type, x_n", [(1, x_n), (2, x_n2)]) # Test using DCT-I and DCT-II
def test_cheb_deriv_accurate_to_6th(dct_type, x_n):
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
	L2_powers = [[-19, -14, -10, -6, -3, 0], [-17, -13, -9, -6, -2, 1]]
	L1_powers = [[-9, -6, -4, -2, -1, 1], [-8, -6, -4, -2, 0, 1]]

	for nu in range(1,7):
		computed = cheb_deriv(y_n, x_n, nu, dct_type=dct_type) # strangely, this can be slightly different (but close) in CI vs local, despite same package versions
		assert np.nanmean((analytic_truth[nu-1] - computed)**2) < 10**L2_powers[dct_type-1][nu-1]
		assert np.nanmax(np.abs(analytic_truth[nu-1] - computed)) < 10**L1_powers[dct_type-1][nu-1]

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
			computed = cheb_deriv(y_n, x_n, nu)
			assert np.abs(computed[0] - analytic_truth[nu-1][0]) < 1e-7
			assert np.abs(computed[-1] - analytic_truth[nu-1][-1]) < 1e-7
		else: # nu > 4
			with pytest.warns(UserWarning): # assure the warning is thrown
				computed = cheb_deriv(y_n, x_n, nu)
			assert np.isnan(computed[0]) # the endpoints are NaN
			assert np.isnan(computed[-1])
			assert np.all(~np.isnan(computed[1:-1])) # the middle isn't NaN
		assert np.nanmean(analytic_truth[nu-1] - computed)**2 < 1e-7 # check middle too for good measure

def test_fourier_deriv_accurate_to_3rd():
	"""A test for derivatives of a periodic function sampled at equispaced points
	"""
	for th_n_ in (th_n, np.arange(0, M+1) * 2*np.pi / (M+1)): # Test for an odd M too!
		y_n = np.cos(th_n_) + 2*np.sin(3*th_n_)
		analytic_truth = [-np.sin(th_n_) + 6*np.cos(3*th_n_),		# 1st
							-np.cos(th_n_) - 18*np.sin(3*th_n_),	# 2nd
							np.sin(th_n_) - 54*np.cos(3*th_n_)]		# 3rd

		for nu in range(1,4):
			computed = fourier_deriv(y_n, th_n_, nu)
			assert np.nanmean((analytic_truth[nu-1] - computed)**2) < 1e-25
			assert np.nanmax(np.abs(analytic_truth[nu-1] - computed)) < 1e-12

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_fourier_antiderivative_0_to_3rd():
	for th_n_ in (th_n, np.arange(0, M+1) * 2*np.pi / (M+1)): # Test for an odd M too!
		y_n = np.cos(th_n_) + 2*np.sin(3*th_n_)
		analytic_truth = [y_n,											# 0th
							np.sin(th_n_) - (2/3)*np.cos(3*th_n_),		# -1st
							-np.cos(th_n_) - (2/9)*np.sin(3*th_n_),		# -2nd
							-np.sin(th_n_) + (2/27)*np.cos(3*th_n_)]	# -3rd

		for nu in range(0,4):
			computed = fourier_deriv(y_n, th_n_, -nu)
			assert np.nanmean((analytic_truth[nu] - computed)**2) < 1e-25
			assert np.nanmax(np.abs(analytic_truth[nu] - computed)) < 1e-12

def test_cheb_multidimensional():
	"""A test for multidimensional derivatives in the aperiodic case
	"""
	X1_n, X2_n = np.meshgrid(x_n, x_n) # a 100 x 100 grid
	y_n = X1_n**2 * np.sin(3/2 * np.pi * X2_n)
	
	# d^2 / dx_1 dx_2
	analytic_truth = 3 * X1_n * np.pi * np.cos(3/2 * np.pi * X2_n)
	computed = cheb_deriv(cheb_deriv(y_n, x_n, 1, axis=0), x_n, 1, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-18
	assert np.max(np.abs(analytic_truth - computed)) < 1e-9
	
	# Laplacian
	analytic_truth = 2 * np.sin(3/2 * np.pi * X2_n) - 9/4 * np.pi**2 * X1_n**2 * np.sin(3/2 * np.pi * X2_n)
	computed = cheb_deriv(y_n, x_n, 2, axis=0) + cheb_deriv(y_n, x_n, 2, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-16
	assert np.max(np.abs(analytic_truth - computed)) < 1e-6

def test_fourier_multidimensional():
	"""A test for multidimensional derivatives in the periodic case
	"""
	T1_n, T2_n = np.meshgrid(th_n, th_n) # a 100 x 100 grid
	y_n = np.sin(2*T1_n) * np.cos(T2_n)

	#d^2 / d theta_1 d theta_2
	analytic_truth = -2 * np.cos(2 * T1_n) * np.sin(T2_n)
	computed = fourier_deriv(fourier_deriv(y_n, th_n, 1, axis=0), th_n, 1, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-25
	assert np.max(np.abs(analytic_truth - computed)) < 1e-14

	# Laplacian
	analytic_truth = -5 * np.sin(2 * T1_n) * np.cos(T2_n)
	computed = fourier_deriv(y_n, th_n, 2, axis=0) + fourier_deriv(y_n, th_n, 2, axis=1)
	assert np.mean((analytic_truth - computed)**2) < 1e-25
	assert np.max(np.abs(analytic_truth - computed)) < 1e-13

def test_cheb_arbitrary_domains_to_3rd():
	"""A test that we can take the derivative on domains that aren't the canonical [1, -1]
	"""
	L2_powers = [[-10, -6, -2],[-25, -24, -20]] # The function does *much* better on shorter 
	L1_powers = [[-4, -2, 1],[-14, -11, -9]] # domains with an N this low and y this wobbly.

	for i,(t_0,t_N) in enumerate([(3.5, 0.5), (-4,-5)]):
		t_n = np.cos(np.arange(N+1) * np.pi / N) * (t_0 - t_N)/2 + (t_0 + t_N)/2

		y_n = np.exp(t_n) * np.sin(5*t_n)
		analytic_truth = [5*np.exp(t_n) * np.cos(5*t_n) + np.exp(t_n) * np.sin(5*t_n),	# 1st
							2*np.exp(t_n) * (5*np.cos(5*t_n) - 12*np.sin(5*t_n)),		# 2nd
							-2*np.exp(t_n) * (37*np.sin(5*t_n) + 55*np.cos(5*t_n))]		# 3rd

		for nu in range(1,4):
			computed = cheb_deriv(y_n, t_n, nu)
			assert np.mean((analytic_truth[nu-1] - computed)**2) < 10**L2_powers[i][nu-1]
			assert np.max(np.abs(analytic_truth[nu-1] - computed)) < 10**L1_powers[i][nu-1]

def test_fourier_arbitrary_domains_to_3rd():
	"""A test that we can take the derivative on domains that aren't the canonical [0, 2pi)
	"""
	for t_0 in [0, 4]:
		t_n = (np.arange(0, M) / M) * 4 + t_0

		y_n = np.cos(np.pi/2 * t_n) + 2*np.sin(3*np.pi/2 * t_n)
		analytic_truth = [3*np.pi*np.cos(3*np.pi/2 * t_n) - np.pi/2 * np.sin(np.pi/2 * t_n),					# 1st 
							-(np.pi**2)/4 * np.cos(np.pi/2 * t_n) - (9*np.pi**2)/2 * np.sin(3*np.pi/2 * t_n),	# 2nd
							(np.pi**3)/8 * np.sin(np.pi/2 * t_n) - (27*np.pi**3)/4 * np.cos(3*np.pi/2 * t_n)]	# 3rd

		for nu in range(1,4):
			computed = fourier_deriv(y_n, t_n, nu)
			assert np.mean((analytic_truth[nu-1] - computed)**2) < 1e-23
			assert np.max(np.abs(analytic_truth[nu-1] - computed)) < 1e-11

def test_user_errors():
	"""A test that helpful errors are thrown when a user goofs up
	"""
	for t_n in [np.cos(np.arange(N+1) * np.pi / N)[::-1] * (4 - 1)/2 + (4 + 1)/2, # t_n ordered low-to-high
				np.linspace(4, 1, N+1),											# t_n not cosine-spaced
				np.cos(np.arange(N) * np.pi / (N-1)) * (4 - 1)/2 + (4 + 1)/2]:	# t_n not the same length as y_n along axis
		with pytest.raises(ValueError):
			cheb_deriv(np.zeros(N+1), t_n, 1)
	for t_n in [np.linspace(8, 4, M),	 	# t_n ordered high-to-low
				np.linspace(4, 8, M-1)]:	# t_n not the same length as y_n along axis
		with pytest.raises(ValueError):
			fourier_deriv(np.zeros(M), t_n, 1)

def test_fourier_filter():
	"""A test that applying a filter helps take noise-resistant derivatives of a noisy periodic function
	"""
	for th_n_ in (th_n, np.arange(0, M+1) * 2*np.pi / (M+1)): # Test for an odd M too!
		y_n_with_noise = np.cos(th_n_) + 2*np.sin(3*th_n_) + 0.1*np.random.randn(*th_n_.shape) # add in some gaussian noise
		analytic_truth = [-np.sin(th_n_) + 6*np.cos(3*th_n_),		# 1st
							-np.cos(th_n_) - 18*np.sin(3*th_n_),	# 2nd
							np.sin(th_n_) - 54*np.cos(3*th_n_)]		# 3rd

		for nu in range(1,4): # Things get less accurate for higher derivatives, so check < 10^f(nu)
			computed_noisy = fourier_deriv(y_n_with_noise, th_n_, nu)
			computed_noisy_with_filter = fourier_deriv(y_n_with_noise, th_n_, nu, filter=lambda k: np.abs(k) < 5) # only keep lower-frequency modes
			assert np.nanmean((analytic_truth[nu-1] - computed_noisy_with_filter)**2) < np.nanmean((analytic_truth[nu-1] - computed_noisy)**2)
			assert np.nanmax(np.abs(analytic_truth[nu-1] - computed_noisy_with_filter)) < np.nanmax(np.abs(analytic_truth[nu-1] - computed_noisy))

@pytest.mark.filterwarnings('ignore::UserWarning') # Not worrying about warnings in this test
def test_cheb_filter():
	"""A test that applying a filter helps take noise-resistant derivatives of a noisy aperiodic function.
	This test can occasionally fail, related to https://github.com/pavelkomarov/spectral-derivatives/issues/14
	"""
	y_n_with_noise = np.exp(x_n) * np.sin(5*x_n) + 0.1*np.random.randn(*x_n.shape)
	analytic_truth = [5*np.exp(x_n) * np.cos(5*x_n) + np.exp(x_n) * np.sin(5*x_n),	# 1st
						2*np.exp(x_n) * (5*np.cos(5*x_n) - 12*np.sin(5*x_n)),		# 2nd
						-2*np.exp(x_n) * (37*np.sin(5*x_n) + 55*np.cos(5*x_n)),		# 3rd
						4*np.exp(x_n) * (119*np.sin(5*x_n) - 120*np.cos(5*x_n)),	# 4th
						4*np.exp(x_n) * (719*np.sin(5*x_n) + 475*np.cos(5*x_n)),	# 5th
						8*np.exp(x_n) * (2035*np.cos(5*x_n) - 828*np.sin(5*x_n))]	# 6th

	for nu in range(1,7):
		computed_noisy = cheb_deriv(y_n_with_noise, x_n, nu)
		computed_noisy_with_filter = cheb_deriv(y_n_with_noise, x_n, nu, filter=lambda k: k < 10)
		assert np.nanmean((analytic_truth[nu-1] - computed_noisy_with_filter)**2) < np.nanmean((analytic_truth[nu-1] - computed_noisy)**2) # I've seen this assertion fail in CI before. Tests based on randomness are handwavy.
		assert np.nanmax(np.abs(analytic_truth[nu-1] - computed_noisy_with_filter)) < np.nanmax(np.abs(analytic_truth[nu-1] - computed_noisy))
