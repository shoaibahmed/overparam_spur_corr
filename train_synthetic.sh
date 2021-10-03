#!/bin/bash

# Train with group reweighting
python run_toy_example.py -N 100 -o results.csv --toy_example_name no_projections --n 3000 --p_correlation 0.9 --mean_causal 1 --var_causal 1 --mean_spurious 1 --var_spurious 0.01 --mean_noise 0 --var_noise 1 --model_type logistic --error_type zero_one --Lambda 1e-09 --seed 0 --model_file model.pkl
