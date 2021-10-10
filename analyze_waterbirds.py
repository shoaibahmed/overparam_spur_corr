import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os, sys, json, argparse
from summary_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_csv', required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    plot_summary(df, ['oversample', 'undersample', 'ERM'], ['reweight', 'subsample', 'ERM'], 'error_vs_overparam.png')
    summarize_results(df, ['oversample', 'undersample', 'ERM'], ['reweight', 'subsample', 'ERM'], 'summary.json')

if __name__=='__main__':
    main()
