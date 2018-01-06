import scipy.stats as sp
import numpy as np
import sys
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import rc
import argparse as ap
import os
from subprocess import call
import pickle

from common_defs import *
import mi_computing_1
import mi_computing_2
import mi_computing_3
import mi_computing_4
import error_computing_1
import plot_functions

"""
The script is intended for Brutus usage. 

USAGE:
With the parameter '--dispatch':
    Has following subparameters
    --name
    --gamma_val_low
    --gamma_val_high
    --gamma_val_step 
    --noise_model
    --number_solutions
    --repetitions_for_error
    --repetitions_for_mi
    --mi_computing_type  (see mi_computing_modules)

    Creates subfolder for experiment and puts experiment.pk with the necessary info

With parameter '--onejob':
    --name
    --jobnumber

    Runs one separate job with given 'jobnumber'

With parameter '--finalize':
    --name
    --gitcommit
    Aggregates all the necessary information into results.pk

With '--plot'
    --name
    --gitcommit
    Plots everything into figures/...eps


So the pipline: first --dispatch (this will create necessary amount of
--onejob jobs), then --finalize
"""

mi_computing_modules = {
    "mi_computing_1": mi_computing_1,
    "mi_computing_2": mi_computing_2,
    "mi_computing_3": mi_computing_3,
    "mi_computing_4": mi_computing_4,
}

error_computing_modules = {
    "error_computing_1": error_computing_1,
}

def dispatch_jobs( dir_name, args ):
    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_setting_filename + os.extsep + pickle_suffix )

    with open( pickle_filename, 'wb' ) as output:
        experiment = CustomObj()
        experiment.name = args.name
        experiment.number_solutions = args.number_solutions
        experiment.gamma_val_low = args.gamma_val_low
        experiment.gamma_val_high = args.gamma_val_high
        experiment.gamma_val_step = args.gamma_val_step
        experiment.noise_model = args.noise_model
        experiment.repetitions_for_error = args.repetitions_for_error
        experiment.repetitions_for_mi = args.repetitions_for_mi
        experiment.mi_computing_type = args.mi_computing_type
        experiment.error_computing_type = args.error_computing_type
        experiment.job_gammas = np.arange( args.gamma_val_low, args.gamma_val_high, 
            args.gamma_val_step )

        # Saving a file with the experiment setting
        pickle.dump( experiment, output )

    # # Dispatching
    # for k in range( len( experiment.job_gammas ) ):
    #     call( ['bsub', '-W', '08:00', '-R', 'rusage[mem=2048]',
    #         'python', 'run_simple-model_mut-inf_cluster.py',
    #         '--onejob', '--jobnumber', str( k ),
    #         '--name', args.name ] )

    # For non-cluster debug purposes
    for k in range( len( experiment.job_gammas ) ):
        call( [ 'python', 'run_simple-model_mut-inf_cluster.py',
            '--onejob', '--jobnumber', str( k ),
            '--name', args.name ] )

def process_onejob( dir_name, args ):
    # Open experiment setting
    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_setting_filename + os.extsep + pickle_suffix )
    with open( pickle_filename, 'rb') as input:
        experiment = pickle.load( input )
    
    gamma = experiment.job_gammas[ args.jobnumber ]
    mi_comp_module = mi_computing_modules[ experiment.mi_computing_type ]
    
    [mutual_inf, p_joint, h_joint, h_single] = \
        mi_comp_module.compute_mutual_inf( gamma, experiment )

    error_computing_module = error_computing_modules[ experiment.error_computing_type ]
    
    [mutual_inf, p_joint, h_joint, h_single] = \
        mi_comp_module.compute_mutual_inf( gamma, experiment )
    
    mean_err, std_err = error_computing_module.compute_error( gamma,
        experiment )

    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_jobresult_filename + '_' + str( args.jobnumber ) + os.extsep 
        + pickle_suffix )
    with open( pickle_filename, 'wb' ) as output:
        onejob_result = CustomObj()
        onejob_result.jobnumber = args.jobnumber
        onejob_result.job_mean_err = mean_err
        onejob_result.job_std_err = std_err
        onejob_result.job_mutual_inf = mutual_inf
        onejob_result.job_p_joint = p_joint
        onejob_result.job_h_joint = h_joint
        onejob_result.job_h_single = h_single
        
        # Saving a file.
        pickle.dump( onejob_result, output )

def finalize_results( dir_name, args ):
    # Open experiment setting
    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_setting_filename + os.extsep + pickle_suffix )
    with open( pickle_filename, 'rb') as input:
        experiment = pickle.load( input )

    # If need to add to git
    if args.gitcommit:
        call( ['git', 'add', pickle_filename ] )

    mutual_inf_final = np.zeros( len( experiment.job_gammas ) ) 
    p_joint_final = np.zeros( ( len( experiment.job_gammas ), 
        experiment.number_solutions, experiment.number_solutions ) ) 
    h_joint_final = np.zeros( len( experiment.job_gammas ) ) 
    h_single_final = np.zeros( len( experiment.job_gammas ) ) 
    mean_err_final = np.zeros( len( experiment.job_gammas ) ) 
    std_err_final = np.zeros( len( experiment.job_gammas ) ) 

    for k in range( len( experiment.job_gammas ) ):
        pickle_filename = os.path.join( dir_name, args.name + '_' 
            + experiment_jobresult_filename + '_' + str( k ) + os.extsep 
            + pickle_suffix )
        with open( pickle_filename, 'rb' ) as input:
            onejob_result = pickle.load( input )
            mutual_inf_final[k] = onejob_result.job_mutual_inf
            p_joint_final[k] = onejob_result.job_p_joint
            h_joint_final[k] = onejob_result.job_h_joint
            h_single_final[k] = onejob_result.job_h_single
            mean_err_final[k] = onejob_result.job_mean_err
            std_err_final[k] = onejob_result.job_std_err

    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_finalresult_filename + os.extsep + pickle_suffix )
    with open( pickle_filename, 'wb' ) as output:
        final_result = CustomObj()
        final_result.mutual_inf_final = mutual_inf_final
        final_result.p_joint_final = p_joint_final
        final_result.h_joint_final = h_joint_final
        final_result.h_single_final = h_single_final
        final_result.mean_err_final = mean_err_final
        final_result.std_err_final = std_err_final

        # Saving a file with the final result.
        pickle.dump( final_result, output )

    if args.gitcommit:
        call( ['git', 'add', pickle_filename ] )
        call( ['git', 'commit', '-m', '"Automatic commit"' ] )
        call( ['git', 'push' ] )

# main script file
if __name__ == "__main__":

    parser = ap.ArgumentParser( description='Brutus script arguments' )
    parser.add_argument( '--dispatch', action='store_true',
            help='Run the script as a dispatcher, needs arguments' )
    parser.add_argument('--name', action='store',
            help='Name of the experiment', required=True)
    parser.add_argument('--number_solutions', action='store', type=int, 
        help='Number of solutions')
    parser.add_argument('--noise_model', action='store', help='Noise model used for it' )
    parser.add_argument('--mi_computing_type', action='store', help='Type of computing '
        'mutual information' )
    parser.add_argument('--error_computing_type', action='store', help='Type of computing '
        'error' )
    parser.add_argument('--onejob', action='store_true', help='Run one job')
    parser.add_argument('--gamma_val_low', action='store', type=int, help='Lower gamma_range')
    parser.add_argument('--gamma_val_high', action='store', type=int, help='Higher gamma_range')
    parser.add_argument('--gamma_val_step', action='store', type=int, help='Step gamma_range')
    parser.add_argument('--jobnumber', action='store', type=int, 
            help='Currently evaluated job number')
    parser.add_argument('--repetitions_for_error', action='store', type=int, 
            help='Number of repetitions to simulate error')
    parser.add_argument('--repetitions_for_mi', action='store', type=int, 
            help='Number of repetitions for computing MI')
    parser.add_argument('--finalize', action='store_true', help='Finalize several jobs results')
    parser.add_argument('--plot', action='store_true', help='Plot results into file')
    parser.add_argument('--gitcommit', action='store_true', help='Whether to commit finalized'
        ' or plotted result' )

    args = parser.parse_args()

    dir_name = os.path.dirname( os.path.abspath( __file__ ) )
    figs_dir_name = os.path.join( dir_name, figures_folder_name ) 
    dir_name = os.path.join( dir_name, experiments_folder_name ) 
    if not os.path.exists( dir_name ):
        os.mkdir( dir_name )
    if not os.path.exists( figs_dir_name ):
        os.mkdir( figs_dir_name )

    dir_name = os.path.join( dir_name, args.name )
    if not os.path.exists( dir_name ):
        os.mkdir( dir_name )

    # Now perform the job.
    if args.dispatch:
        dispatch_jobs( dir_name, args )
    
    elif args.onejob:
        process_onejob( dir_name, args )

    elif args.finalize:
        finalize_results( dir_name, args )

    elif args.plot:
        plot_functions.plot_mutual_information( dir_name, figs_dir_name, args )

