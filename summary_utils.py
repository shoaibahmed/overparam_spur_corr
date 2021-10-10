import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os, sys, json

def plot_error_vs_overparam(fig, ax, results_df, opt_type, robust=True, train=True,
                            color='dark grey', marker='.', verbose=False):
    field = ''
    if robust:
        field+='robust'
    else:
        field+='avg'
    if train:
        field+='_train'
    else:
        field+='_test'
    field+='_error'
    
    selected_results = results_df[results_df['opt_type']==opt_type]
    ax.scatter(selected_results['N'], selected_results[field],
               color=color, marker=marker, alpha=0.5)
    mean_results = selected_results[['N', field]].groupby('N').mean().reset_index().sort_values('N')
    mean_curve = ax.semilogx(mean_results['N'], mean_results[field],
                             color=color, alpha=0.8)
    return mean_curve

def plot_double_descent_standard(results_df, xlabel, opt_types,
                                 fig=None, ax=None,
                                 color_dict={(True, True): 'lightblue', (True, False):'tab:blue',
                                             (False, True): 'darkgray', (False, False):'black'},
                                 verbose=True):
    plotted_curves = {}
    if ax is None:
        fig, ax = plt.subplots(1, len(opt_types), figsize=(len(opt_types)*5,3), sharex=True, sharey='row')
    for i, opt_type in enumerate(opt_types):
        for robust in [False, True]:
            for train in [True, False]:
                curve, = plot_error_vs_overparam(fig, ax[i], results_df, opt_type,
                                                robust=robust, train=train, color=color_dict[(robust, train)],
                                                verbose=verbose)
                ax[i].set_xlabel(xlabel)
                plotted_curves[(opt_type, robust, train)] = curve
    ax[0].set_ylabel('Error')
    return fig, ax, plotted_curves

def plot_summary(df, opt_types, opt_type_names, outpath):
    plt.rcParams.update({'font.size': 20, 'lines.linewidth':4, 'lines.markersize':12, 
                         'xtick.labelsize':15, 'ytick.labelsize':15, 'axes.labelsize':20,
                         'axes.titlesize': 20, 'legend.fontsize':20,
                         'pdf.fonttype': 42, 'ps.fonttype':42})
    fig, ax = plt.subplots(1, len(opt_types),
                           figsize=(16, 4.5), 
                           sharex=False, sharey=False)
    plot_double_descent_standard(df, 'Number of Random Features', opt_types,
                                 fig=fig, ax=ax)
    for i, opt_name in enumerate(opt_type_names):
        ax[i].set_title(opt_name)
    fig.tight_layout()
    plt.ylim([0,1])
    plt.savefig(outpath, dpi=fig.dpi)


def summarize_results(df, opt_types, opt_type_names, outpath):
    groupby_fields = ['opt_type', 'N']
    metric_fields = ['avg_train_error', 'avg_test_error', 'robust_train_error', 'robust_test_error']
    mean_df = df[groupby_fields + metric_fields].groupby(groupby_fields).mean().reset_index()
    summary = {}
    for opt_type, opt_name in zip(opt_types, opt_type_names):
        summary[opt_name] = {}
        for criterion in ['avg_test_error', 'robust_test_error']:
            summary[opt_name][criterion] = {}
            selected_results = mean_df[mean_df['opt_type']==opt_type]
            idx = np.argmin(selected_results[criterion].values)
            for field in ['N'] + metric_fields:
                summary[opt_name][criterion][field] = selected_results.iloc[idx][field].item()
    with open(outpath, 'w') as f:
        json.dump(summary, f)
    return summary
