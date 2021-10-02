#!/bin/bash

# python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32 --loss_fn ce --log_dir ./logs/
python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32 --loss_fn supcon --log_dir ./logs_supcon/
# python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32 --loss_fn ce --log_dir ./logs_supcon/
