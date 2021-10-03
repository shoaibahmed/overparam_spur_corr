#!/bin/bash

# python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32 --loss_fn ce --log_dir ./logs/
# python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32 --loss_fn ce --log_dir ./logs_supcon/

# Train without group reweighting since the objective should remain the same for SupCon
# python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --train_from_scratch --resnet_width 32 --loss_fn supcon --log_dir ./logs_supcon/

# Train with group reweighting
# python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width 32 --loss_fn supcon --log_dir ./logs_supcon_group_reweight/

for width in 1 2 4 8 16 32 64 128; do
    # Train using CE loss
    python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn ce --log_dir ./output/logs_ce_r10_w${width}/

    # Train using SupCon loss
    python run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --log_dir ./output/logs_supcon_r10_w${width}/
done
