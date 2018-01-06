import numpy as np
import math
import noise_models
from itertools import izip, product

'''
Various MI computation functions needed in the experiment.
The MI is computed as MI between boolean vectors concidered as histograms.
See notes for more.
'''
	
'''
Computes classical mutual information.
'''
def compute_mutual_inf( gamma, experiment ):
	
	number_solutions = experiment.number_solutions
	noise_model = experiment.noise_model
	repetitions_for_mi = experiment.repetitions_for_mi

	p_joint = np.zeros( ( number_solutions, number_solutions ) )
	p_single = np.zeros( number_solutions )

	for k in range( repetitions_for_mi ):
		# Generate pair x1, x2 and pair of approximation sets
		[x1, x2] = noise_models.generate_pair( number_solutions, noise_model )
		as1 = max( x1 - gamma, 0 ), min( x1 + gamma + 1, number_solutions )
		as2 = max( x2 - gamma, 0 ), min( x2 + gamma + 1, number_solutions )
		len1 = as1[1] - as1[0]
		len2 = as2[1] - as2[0]

		p_joint[ as1[0]:as1[1], as2[0]:as2[1] ] += 1.0/( len1 * len2 )
		p_single[ as1[0]:as1[1] ] += 0.5 * 1.0/( len1 )
		p_single[ as2[0]:as2[1] ] += 0.5 * 1.0/( len2 )

	p_joint = p_joint / np.sum( p_joint )
	p_single = p_single / np.sum( p_single )

	mutual_inf = 0.0
	
	h_joint = 0.0
	h_single = 0.0

	for (i, j) in product( xrange( number_solutions ), repeat = 2 ):
		if np.allclose( p_joint[i,j], 0 ):
			continue

		# Mutual Inf
		mutual_inf += p_joint[i,j] * math.log( 
			p_joint[i,j]/( p_single[i] * p_single[j] ), 2 )
		
		# Entropy
		h_joint += - p_joint[i,j] * math.log( p_joint[i,j], 2 )
		
		if not np.allclose( p_single[i], 0 ):
			h_single += -p_joint[i,j] * math.log( p_single[i], 2 )

	return mutual_inf, p_joint, h_joint, h_single
