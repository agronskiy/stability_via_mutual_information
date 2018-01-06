import numpy as np
import scipy.stats
from scipy.stats import rv_discrete
import math
from random import randrange

'''
Various functions devoted to noise modelling of the experiment. 
So far the noise is generally <mu> + some_noise

mu is distributed uniformly

'some_noise' is defined by string <noise_model>, which has format
	trunc-gauss_<sigma>
		truncated gaussian with sigma
	one-peaked-debug_<mu_0>
		determenistic: all prob concetrated at mu_0, which is fixed itself
	one-peaked-running-debug
		mu_0 is running over solutions uniformly, both x' and x'' are in mu_0
'''

trunc_gauss_nm = 'trunc-gauss'
one_peaked_debug_nm = 'one-peaked-debug'
one_peaked_running_debug_nm = 'one-peaked-running-debug'
two_independent_solutions_nm = 'two-independent'

def p_mu( mu, noise_model, number_solutions ):
	noise_model_list = noise_model.split( '_' )
	if noise_model_list[0] == trunc_gauss_nm:
		return 1.0/number_solutions

	if noise_model_list[0] == one_peaked_debug_nm:
		mu_0 = float( noise_model_list[1] )
		if np.allclose( mu, 0 ):
			return 1.0
		else:
			return 0.0

	if noise_model_list[0] == one_peaked_running_debug_nm:
		return 1.0/number_solutions

	if noise_model_list[0] == two_independent_solutions_nm:
		return 1.0/number_solutions

'''
Returns 0 outside of solution space
'''
def p_x_given_mu( x, mu, noise_model, number_solutions ):
	noise_model_list = noise_model.split( '_' )
	
	if noise_model_list[0] == trunc_gauss_nm:
		sigma = float( noise_model_list[1] )
		left, right = 0, number_solutions - 1
		left_truncated = (left - mu) / sigma
		right_truncated = (right - mu) / sigma

		# Take into account that we need to discretize, hence using
		#  difference of two cdf's
		return ( scipy.stats.truncnorm.cdf( x + 0.5, left_truncated, right_truncated,
			loc = mu, scale = sigma )  
			- scipy.stats.truncnorm.cdf( x - 0.5, left_truncated, right_truncated,
			loc = mu, scale = sigma ) )

	if noise_model_list[0] == one_peaked_debug_nm:
		if np.allclose( x, mu ):
			return 1.0
		else:
			return 0.0

	if noise_model_list[0] == one_peaked_running_debug_nm:
		if np.allclose( x, mu ):
			return 1.0
		else:
			return 0.0

	if noise_model_list[0] == two_independent_solutions_nm:
		return 1.0/number_solutions

def generate_pair( number_solutions, noise_model ):
	mu = randrange( number_solutions )

	xk = np.arange( number_solutions )
	pk = map( lambda x: p_x_given_mu( x, mu, noise_model, 
		number_solutions ), xk )
	
	rv = rv_discrete( name='custm', values = ( xk, pk ) )

	return ( rv.rvs(), rv.rvs(), mu )