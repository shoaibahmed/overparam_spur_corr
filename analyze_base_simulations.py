import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os, sys, json, argparse
from summary_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_results_csv', required=True)
    parser.add_argument('--no_spu_results_csv', required=True)
    args = parser.parse_args()

    base_df = pd.read_csv(args.base_results_csv)
    no_spu_df = pd.read_csv(args.no_spu_results_csv)
    plot_summary(base_df, ['oversample', 'undersample', 'ERM'], ['reweight', 'subsample', 'ERM'], 'error_vs_overparam.png')
    summarize_results(base_df, ['oversample', 'undersample', 'ERM'], ['reweight', 'subsample', 'ERM'], 'summary.json')
    figure_toy_intuition(base_df, no_spu_df, 'intuition.png')

def figure_toy_intuition(base_df, no_spu_df, outpath):
    plt.rcParams.update({'font.size': 20, 'lines.linewidth':4, 'lines.markersize':12, 
                         'xtick.labelsize':18, 'ytick.labelsize':15, 'axes.labelsize':18,
                         'axes.titlesize': 20, 'hatch.linewidth':1, 'legend.fontsize':18,
                         'axes.ymargin':-0})
    plt.set_cmap('tab20')
    # Figure
    fig, ax = plt.subplots(1, 2,  
                           figsize=(12,4.5), sharex=False, sharey='row')
    
    #Plot
    dfs = [no_spu_df, base_df]
    labels = ['core only', 'core and spurious']
    curves = []
    for i, train in enumerate([True, False]):
        for j, df in enumerate(dfs):
            curve, = plot_error_vs_overparam(fig, ax[0],
                                             df, 'oversample',
                                             robust=True,
                                             train=train,
                                             color=plt.cm.get_cmap('tab20').colors[((j+2)%3)*2+((i+1)%2)])
            if not train:
                curves.append(curve)
    # Legend
    reorder = [1,0]
    legend_curves = [curves[i] for i in reorder] 
    legend_labels = [labels[i] for i in reorder] 
    ax[0].legend(handles=legend_curves, labels=legend_labels, ncol=1, loc='upper right')
    ax[0].set_ylabel('Error')
    ax[0].set_xlabel('Parameter Count')
    
    # Bar plot
    for i, overparam in enumerate([80,10000]):
        for j, groups in enumerate([[0,3],[1,2]]):
            idx = np.arange(i*5,i*5+4)
            stats = base_df[(base_df['N']==overparam) & (base_df['opt_type']=='oversample')].mean()
            train_errors = [stats[f'train_error_group:{g}'] for g in range(4)]
            test_errors = [stats[f'test_error_group:{g}'] for g in range(4)]
            test_bar = ax[1].bar(idx, test_errors, 
                              color = ['dimgrey',plt.cm.get_cmap('tab20').colors[0],
                                       plt.cm.get_cmap('tab20').colors[0], 'dimgrey'],
                              edgecolor=['dimgrey',plt.cm.get_cmap('tab20').colors[0],
                                         plt.cm.get_cmap('tab20').colors[0], 'dimgrey'],
                              linewidth=1)
            train_bar = ax[1].bar(idx, train_errors, hatch='/////',
                               color = ['dimgrey',plt.cm.get_cmap('tab20').colors[0],
                                       plt.cm.get_cmap('tab20').colors[0],'dimgrey',],
                               edgecolor=['silver',plt.cm.get_cmap('tab20').colors[1],
                                         plt.cm.get_cmap('tab20').colors[1],'silver',],
                               linewidth=1)
    ax[1].set_xticks([1.5,6.5])
    ax[1].set_xticklabels(['Underparameterized\n(m=80)', 'Overparameterized\n(m=10000)' ])
    ax[1].legend((train_bar[1], test_bar[1], test_bar[1], test_bar[0]), 
              ('train','test',r'minority ($y\neq a$)',r'majority($y=a$)'),
               ncol=2)
    ax[1].set_ylim([0,1])
    fig.tight_layout()
    plt.savefig(outpath)

if __name__=='__main__':
    main()
