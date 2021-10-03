#!/bin/bash

for width in 1 2 4 8 16 32 64 128; do
    # Train using CE loss
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=50G \
                  --kill-on-bad-exit --job-name celebA-ce-w${width} --nice=0 \
                  --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.05-py3.sqsh \
                  --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --scheduler \
                    --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 100 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn ce \
                    --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output/logs_ce_r10_w${width}/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output/logs_ce_r10_w${width}.log 2>&1 &

    # Train using SupCon loss
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=50G \
                  --kill-on-bad-exit --job-name celebA-supcon-w${width} --nice=0 \
                  --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.05-py3.sqsh \
                  --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 --scheduler \
                    --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 100 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon \
                    --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output/logs_supcon_r10_w${width}/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output/logs_supcon_r10_w${width}.log 2>&1 &
done
