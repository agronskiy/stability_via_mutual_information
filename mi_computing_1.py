import numpy as np
import math
import noise_models
from itertools import izip

'''
Various computation functions needed in the experiment.
'''

'''
Computes p( C_\gamma | \mu).
'''
def p_approx_set_given_mu( approx_set, gamma, mu, number_solutions, noise_model ):
	# approx_set: a pair (begin, end).
	begin, end = approx_set
	centers_range = [0, number_solutions]
	
	if begin > 0:
		centers_range[0] = max( centers_range[0], begin + gamma )
	if end < number_solutions:
		centers_range[1] = min( centers_range[1], end - gamma )
	if begin == 0:
		centers_range[1] = min( centers_range[1], begin + gamma )
	if end == number_solutions:
		centers_range[0] = max( centers_range[0], end - gamma )

	# centers_range contains (left, right) possible positions of the approx set
	#  center. 
	if centers_range[0] >= centers_range[1]:
		return 0.0
	else:
		return sum( map( lambda x: noise_models.p_x_given_mu( x, mu, noise_model, 
			number_solutions ), range( centers_range[0], centers_range[1] ) ) )

'''
Computes one term of classical mutual information sum.
'''
def compute_one_mutual_inf_term( gamma, experiment, approx_set1, approx_set2, 
		noise_model ):
	number_solutions = experiment.number_solutions

	p_12 = 0.0
	p_1 = 0.0
	p_2 = 0.0

	# Here computing integrals (sums for discrete case).
	# TODO: add hashing here.
	for mu in range( number_solutions ):
		curr_p_1_mu = p_approx_set_given_mu( approx_set1, gamma, mu, 
			number_solutions, noise_model )
		curr_p_2_mu = p_approx_set_given_mu( approx_set2, gamma, mu, 
			number_solutions, noise_model )
		curr_p_mu = noise_models.p_mu( mu, noise_model, number_solutions ) 
		
		p_12 += curr_p_mu * curr_p_1_mu * curr_p_2_mu
		p_1 += curr_p_mu * curr_p_1_mu
		p_2 += curr_p_mu * curr_p_2_mu

	# For the case of log(0 / ...) in the definition of mutual information zero
	#	is used.
	if np.allclose( p_12, 0 ):
		return 0

	return p_12 * math.log( p_12/(p_1 * p_2), 2 )

'''
Computes classical mutual information as vectors.
'''
def compute_mutual_inf( gamma, experiment ):
	number_solutions = experiment.number_solutions
	mutual_inf = 0.0
	for approx_set1 in izip( *np.triu_indices( number_solutions ) ): 
		for approx_set2 in izip( *np.triu_indices( number_solutions ) ): 
			mutual_inf += compute_one_mutual_inf_term( 
				gamma, experiment, approx_set1, approx_set2, experiment.noise_model )
	return mutual_inf
