import numpy as np
import noise_models

'''
Computing error by sampling
'''
	
'''
Computes error. Returns average error and interval.
'''
def compute_error( gamma, experiment ):
	
	number_solutions = experiment.number_solutions
	noise_model = experiment.noise_model
	repetitions_for_error = experiment.repetitions_for_error

	errors = np.zeros( repetitions_for_error )

	for k in range( repetitions_for_error ):
		# Generate pair x and respective approx set.
		[x, foo, true_mu] = noise_models.generate_pair( number_solutions, noise_model )
		approx_set = max( x - gamma, 0 ), min( x + gamma + 1, number_solutions )
		errors[k] = abs( np.random.randint( approx_set[0], high = approx_set[1] ) 
			- true_mu )

	return np.mean( errors ), np.std( errors )


