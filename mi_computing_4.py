import numpy as np
import math
import noise_models
from itertools import izip, product

'''
Various MI computation functions needed in the experiment.
The MI is computed as MI between solution sampled from approx set and 
the input mu
'''
	
'''
Computes classical mutual information.
'''
def compute_mutual_inf( gamma, experiment ):
	
	number_solutions = experiment.number_solutions
	noise_model = experiment.noise_model

	p_joint = np.zeros( ( number_solutions, number_solutions ) )

	for (i, mu) in product( xrange( number_solutions ), repeat = 2 ):
		possible_x = xrange( max( i - gamma, 0 ), min( i + gamma + 1, number_solutions ) )

		p_i_given_mu = 0.0
		for x in possible_x:
			# Need to compute the lengths of respective approx sets
			approx_set = max( x - gamma, 0 ), min( x + gamma + 1, number_solutions )
			len_as = approx_set[1] - approx_set[0]

			p_i_given_mu += 1.0/len_as * noise_models.p_x_given_mu( x, mu, 
				noise_model, number_solutions )

		p_joint[i, mu] = ( noise_models.p_mu( mu, noise_model, number_solutions ) 
			* p_i_given_mu )

	p_single_i = np.sum( p_joint, axis = 1 )
	p_single_mu = np.sum( p_joint, axis = 0 )

	mutual_inf = 0.0

	for (i, mu) in product( xrange( number_solutions ), repeat = 2 ):
		if np.allclose( p_joint[i,mu], 0 ):
			continue

		# Mutual Inf
		mutual_inf += p_joint[i,mu] * math.log( 
			p_joint[i,mu]/( p_single_i[i] * p_single_mu[mu] ), 2 )

	return mutual_inf, p_joint, 0.0, 0.0
