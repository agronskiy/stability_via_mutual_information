import numpy as np
import math
import noise_models
from itertools import izip, product
import matplotlib

matplotlib.use( 'MacOSX' )

from matplotlib import pyplot as plt


number_solutions = 20
noise_model = 'trunc-gauss_5'

gamma = 10

p_joint = np.zeros( ( number_solutions, number_solutions ) )

for (i, j) in izip( *np.triu_indices( number_solutions ) ):

	possible_x1 = xrange( max( i - gamma, 0 ), min( i + gamma + 1, number_solutions ) )
	possible_x2 = xrange( max( j - gamma, 0 ), min( j + gamma + 1, number_solutions ) )


	for (x1, x2) in product( possible_x1, possible_x2 ):

		as1 = max( x1 - gamma, 0 ), min( x1 + gamma + 1, number_solutions )
		as2 = max( x2 - gamma, 0 ), min( x2 + gamma + 1, number_solutions )
		len1 = as1[1] - as1[0]
		len2 = as2[1] - as2[0]

		p_x1_x2 = 0.0
		for mu in xrange( number_solutions ):
			p_x1_x2 += ( noise_models.p_mu( mu, noise_model, number_solutions ) 
				* noise_models.p_x_given_mu( x1, mu, noise_model, number_solutions )
				* noise_models.p_x_given_mu( x2, mu, noise_model, number_solutions ) )

		p_joint[i,j] += ( 1.0 / ( len1 * len2 )
			* p_x1_x2 )
		p_joint[j,i] = p_joint[i,j]

p_single = np.sum( p_joint, axis = 0 )

print p_joint
print np.sum( p_joint )

mutual_inf = 0.0

for (i, j) in product( xrange( number_solutions ), repeat = 2 ):
	if np.allclose( p_joint[i,j], 0 ):
		continue

	# Mutual Inf
	mutual_inf += p_joint[i,j] * math.log( 
		p_joint[i,j]/( p_single[i] * p_single[j] ), 2 )
	
	# # Entropy
	# h_joint += - p_joint[i,j] * math.log( p_joint[i,j], 2 )
	
	# if not np.allclose( p_single[i], 0 ):
	# 	h_single += -p_joint[i,j] * math.log( p_single[i], 2 )

print mutual_inf

# plt.ioff()

# noise_model = 'trunc-gauss_1'
# number_solutions = 30

# fig = plt.figure()
# ax = fig.add_subplot(111)

# gamma = 15

# xr = np.arange( 0, number_solutions )
# yr = np.zeros( number_solutions )

# for i in xr:
# 	approx_set = range( max( i - gamma, 0 ), min( i + gamma + 1, number_solutions ) )
# 	for x in approx_set:
# 		for mu in xr:
# 			yr[x] += 1.0 / len( approx_set ) \
# 				* noise_models.p_mu( mu, noise_model, number_solutions ) \
# 				* noise_models.p_x_given_mu( x, mu, noise_model, 
# 				number_solutions )

# plt.plot( xr, yr )
# plt.draw()
# plt.waitforbuttonpress()
# plt.close()
