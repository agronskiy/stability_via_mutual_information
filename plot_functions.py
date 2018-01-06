import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import os
from subprocess import call
import pickle

from common_defs import *

title_font = 22
legend_font = 22
axis_font = 25
ticks_font = 22
line_width = 2.0

rc('text', usetex=True)
rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

def plot_mutual_information( dir_name, figures_dir_name, args ):
    # Open experiment setting
    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_setting_filename + os.extsep + pickle_suffix )
    with open( pickle_filename, 'rb') as input:
        experiment = pickle.load( input )

    # Open results
    pickle_filename = os.path.join( dir_name, args.name + '_' 
        + experiment_finalresult_filename + os.extsep + pickle_suffix )
    with open( pickle_filename, 'rb' ) as input:
        final_result = pickle.load( input )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title_template = r'name: {name}' '\n' r'noise model: {noisemod}' '\n' \
        r'solutions: {solutions}; $\gamma$ values:{gammalow}-{gammahigh}' '\n' \
        r'MI type: {mi_computing_type}'
    title_template = title_template.format( 
        name = experiment.name.replace('_', r'\_'), 
        noisemod = experiment.noise_model.replace('_', r'\_'),
        solutions = str( experiment.number_solutions ),
        gammalow = str( experiment.gamma_val_low ),
        gammahigh = str( experiment.gamma_val_high ), 
        mi_computing_type = experiment.mi_computing_type.replace('_', r'\_') )
    
    ax.set_title( title_template, fontsize=title_font )

    plt.plot( experiment.job_gammas, final_result.mutual_inf_final,
        label = r'theoretical MI' )
    # plt.plot( experiment.job_gammas, final_result.h_joint_final,
    #     label = r'joint entropy' )
    # plt.plot( experiment.job_gammas, final_result.h_single_final,
    #     label = r'single entropy' )

    # Legend tuning.
    legend = ax.legend(loc=1, fontsize=legend_font)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#cccccc')
    
    for legobj in legend.legendHandles:
        legobj.set_linewidth(line_width)

    # Labels tuning
    plt.xlabel(r'$\boldsymbol{\gamma}$', fontsize=axis_font)
    plt.ylabel(r"$I(C'_{\gamma}, C''_{\gamma})$", fontsize=axis_font)
    plt.setp(ax.get_xticklabels(), fontsize=ticks_font)
    plt.setp(ax.get_yticklabels(), fontsize=ticks_font)
    
    plt.tight_layout()
    
    figure_filename = '{name}__{noisemod}__sol{solutions}__gamma{gammalow}-{gammahigh}.eps'
    figure_filename = figure_filename.format( name = experiment.name, 
        noisemod = experiment.noise_model,
        solutions = str( experiment.number_solutions ),
        gammalow = str( experiment.gamma_val_low ),
        gammahigh = str( experiment.gamma_val_high ) )

    figure_filename = os.path.join( figures_dir_name, figure_filename )
    plt.savefig( figure_filename, format='eps', dpi=600 )
    plt.close()

    if args.gitcommit:  
        call( ['git', 'add', figure_filename ] )

    # ###########################################################
    
    for k in range( len( experiment.job_gammas ) ):
        fig = plt.figure()
        ax = fig.add_subplot( 1, 1, 1, aspect='equal' )
        title_template = r'name: {name}' '\n' r'noise model: {noisemod}' '\n' \
            r'solutions: {solutions}; $\gamma$ value:{gammaval}' '\n' \
            r'MI type: {mi_computing_type}'
        title_template = title_template.format( 
            name = experiment.name.replace('_', r'\_'), 
            noisemod = experiment.noise_model.replace('_', r'\_'),
            solutions = str( experiment.number_solutions ),
            gammaval= str( experiment.job_gammas[k] ), 
            mi_computing_type = experiment.mi_computing_type.replace('_', r'\_') )
        
        ax.set_title( title_template, fontsize=title_font )
        plt.pcolor( final_result.p_joint_final[ experiment.job_gammas[k] ] )
        plt.colorbar()

        plt.tight_layout()

        figure_filename = '{name}__p_joint__gamma{gamma}.eps'
        figure_filename = figure_filename.format( name = experiment.name, 
            noisemod = experiment.noise_model,
            gamma = str( experiment.job_gammas[k] ) )

        figure_filename = os.path.join( figures_dir_name, figure_filename )
        plt.savefig( figure_filename, format='eps', dpi=600 )
        plt.close()

        if args.gitcommit:  
            call( ['git', 'add', figure_filename ] )

    # #############################################################

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title_template = r'name: {name}' '\n' r'noise model: {noisemod}' '\n' \
        r'solutions: {solutions}; $\gamma$ values:{gammalow}-{gammahigh}' '\n' \
        r'Error type: {error_computing_type}'
    title_template = title_template.format( 
        name = experiment.name.replace('_', r'\_'), 
        noisemod = experiment.noise_model.replace('_', r'\_'),
        solutions = str( experiment.number_solutions ),
        gammalow = str( experiment.gamma_val_low ),
        gammahigh = str( experiment.gamma_val_high ), 
        error_computing_type = experiment.error_computing_type.replace('_', r'\_') )
    
    ax.set_title( title_template, fontsize=title_font )

    plt.errorbar( experiment.job_gammas, final_result.mean_err_final,
        final_result.std_err_final, label = r'Error', ecolor = 'black', marker = 'o' )

    # Legend tuning.
    legend = ax.legend(loc=1, fontsize=legend_font)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('#cccccc')
    
    for legobj in legend.legendHandles:
        legobj.set_linewidth(line_width)

    # Labels tuning
    plt.xlabel(r'$\boldsymbol{\gamma}$', fontsize=axis_font)
    plt.ylabel(r"Error", fontsize=axis_font)
    plt.setp(ax.get_xticklabels(), fontsize=ticks_font)
    plt.setp(ax.get_yticklabels(), fontsize=ticks_font)
    
    plt.xlim( experiment.job_gammas[0] - 0.2, experiment.job_gammas[-1] + 0.2 )

    plt.tight_layout()
    
    figure_filename = '{name}__err__{noisemod}__sol{solutions}__gamma{gammalow}-{gammahigh}.eps'
    figure_filename = figure_filename.format( name = experiment.name, 
        noisemod = experiment.noise_model,
        solutions = str( experiment.number_solutions ),
        gammalow = str( experiment.gamma_val_low ),
        gammahigh = str( experiment.gamma_val_high ) )

    figure_filename = os.path.join( figures_dir_name, figure_filename )
    plt.savefig( figure_filename, format='eps', dpi=600 )
    plt.close()

    if args.gitcommit:  
        call( ['git', 'add', figure_filename ] )

    # #############################################################
    if args.gitcommit:  
        call( ['git', 'commit', '-m', '"Automatic commit"' ] )
        call( ['git', 'push' ] )


