import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os, sys, json, argparse
from summary_utils import plot_double_descent_standard, plot_error_vs_overparam, summarize_results

def main():
    df = load_results()
    plot_summary(df, ['reweight', 'subsample', 'ERM'], ['reweight', 'subsample', 'ERM'], 'error_vs_overparam.png')
    summarize_results(df, ['reweight', 'subsample', 'ERM'], ['reweight', 'subsample', 'ERM'], 'summary.json')

def load_results(seeds=[0,1,2],
                 # widths=[1,2,4,6,8,16,32,48,64,80,88,96,128,144,160,176,192,224],
                 widths=[1,2,4,6,8,16,32,48,64,80,88,96],
                 epochs={'ERM':49, 'reweight':49, 'subsample':499},
                 splits=['train','val','test'],
                 loss_funcs=['ce', 'supcon'],
                 n_groups=4,
                 smooth_val_window=10):
    # helpers
    def get_dirpath(opt_type, width, seed, loss_func):
        # rundir = f"output_{loss_func}_{opt_type.lower()}_{'lambda_0.01_' if loss_func == 'center' else ''}cosine_augment_fixed_proj_128/celebA_{opt_type.lower()}_width_{width}_seed_{seed}_{loss_func}"
        rundir = f"output_{loss_func}_{opt_type.lower()}_{'lambda_0.01_' if loss_func == 'center' else 'lambda_0.1_' if loss_func == 'distill' else ''}cosine_augment/celebA_{opt_type.lower()}_width_{width}_seed_{seed}_{loss_func}"
        linear_eval_dir = os.path.join(rundir, "linear_eval")
        if loss_func == 'supcon' and os.path.exists(linear_eval_dir):
            rundir = linear_eval_dir
        # rundir = f'output_{loss_func}_reweight/celebA_{opt_type}_width_{width}_seed_{seed}_{loss_func}'
        return rundir

    # columns
    group_columns = [f'avg_acc_group:{g}' for g in range(4)]
    robust_column = 'robust_acc'
    avg_column = 'avg_acc'

    #initialization
    results = []
    # for loss_func in loss_funcs:
    for loss_func in ['distill']:
        for opt_type, last_epoch in epochs.items():
            if loss_func == 'supcon':
                last_epoch = 9
            for width in widths:
                for seed in seeds:
                    row = {}
                    row['opt_type'] = opt_type
                    row['N'] = width
                    row['seed'] = seed
                    for split in splits:
                        # paths
                        rundir = get_dirpath(opt_type, width, seed, loss_func)
                        log_path = os.path.join(rundir, f"{split}{'_linear' if loss_func == 'supcon' else ''}.csv")
                        # skip if we haven't run it yet
                        if not os.path.exists(log_path):
                            print(f'{log_path} not found')
                            continue
                        #  
                        df = pd.read_csv(log_path)
                        df[robust_column] = df[group_columns].min(axis=1)
                        last_epoch_row = df[(df['epoch']<=last_epoch) & (df['epoch']>(last_epoch-smooth_val_window))]
                        if split=='train':
                            row['avg_train_error'] = 1-last_epoch_row[avg_column].values.mean()
                            row['robust_train_error'] = 1-last_epoch_row[robust_column].values.mean()
                            for g in range(4):
                                row[f'train_error_group:{g}'] = 1-last_epoch_row[f'avg_acc_group:{g}'].values.mean()
                        elif split=='test':
                            row['avg_test_error'] = 1-last_epoch_row[avg_column].values.mean()
                            row['robust_test_error'] = 1-last_epoch_row[robust_column].values.mean()
                            for g in range(4):
                                row[f'test_error_group:{g}'] = 1-last_epoch_row[f'avg_acc_group:{g}'].values.mean()
                    results.append(row)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.csv', index=False)
    return results_df

def plot_summary(df, opt_types, opt_type_names, outpath):
    plt.rcParams.update({'font.size': 20, 'lines.linewidth':4, 'lines.markersize':12, 
                         'xtick.labelsize':15, 'ytick.labelsize':15, 'axes.labelsize':20,
                         'axes.titlesize': 20, 'legend.fontsize':20,
                         'pdf.fonttype': 42, 'ps.fonttype':42})
    fig, ax = plt.subplots(1, len(opt_types),
                           figsize=(16, 4.5), 
                           sharex=True, sharey=True)
    plot_double_descent_standard(df, 'ResNet Width', opt_types,
                                 fig=fig, ax=ax)
    for i, opt_name in enumerate(opt_type_names):
        ax[i].set_title(opt_name)
    fig.tight_layout()
    plt.ylim([0,1])
    plt.savefig(outpath, dpi=fig.dpi)


if __name__=='__main__':
    main()
