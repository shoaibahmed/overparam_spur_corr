#!/bin/bash

# for width in 1 2 4 8 16 32 64 128; do
#     # # Train using CE loss
#     srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#                   --kill-on-bad-exit --job-name celebA-ce-w${width} --nice=0 \
#                   --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#                   --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#                 /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 \
#                     --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn ce \
#                     --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_new/logs_ce_r10_w${width}/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_new/logs_ce_r10_w${width}.log 2>&1 &

#     # Train using SupCon loss
#     srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#                   --kill-on-bad-exit --job-name celebA-supcon-w${width} --nice=0 \
#                   --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#                   --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#                 /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.01 \
#                     --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon \
#                     --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_new/logs_supcon_r10_w${width}/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_new/logs_supcon_r10_w${width}.log 2>&1 &
# done

# for seed in 0 1 2; do
for seed in 0; do
    # for width in 1 2 4 6 8 16 32 48 64 80 88 96; do
    # for width in 128 144 160 176 192 224; do
    # for width in 1 2 4 6 8 16 32 48 64 80 88 96 128 144 160 176 192 224; do
    for width in 8 16 32 64 128; do
        # Train using CE loss
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-reweight-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn ce \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_reweight/celebA_reweight_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_reweight/celebA_reweight_width_${width}_seed_${seed}_ce.log 2>&1 &

        # Train using CE loss, and subsampling
        # srun -p RTX6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-subsample-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 500 --save_step 10000 --save_last --log_every 50 --subsample_to_minority --train_from_scratch --resnet_width ${width} --loss_fn ce \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample/celebA_subsample_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample/celebA_subsample_width_${width}_seed_${seed}_ce.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-subsample-w_${width}-s_${seed}-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 500 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --subsample_to_minority --train_from_scratch --resnet_width ${width} --loss_fn "ce" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample_cosine_augment/celebA_subsample_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample_cosine_augment/celebA_subsample_width_${width}_seed_${seed}_ce.log 2>&1 &
        
        # # Train using CE loss, simple ERM (appendix)
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-erm-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --train_from_scratch --resnet_width ${width} --loss_fn ce \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_erm/celebA_erm_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_erm/celebA_erm_width_${width}_seed_${seed}_ce.log 2>&1 &

        # Train using SupCon loss, with reweighting
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed}-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_cosine_aug/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_cosine_aug/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # Train using SupCon loss, with reweighting and large LR
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed}_10cls_e --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 1. \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_large_lr_10cls_e/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_large_lr_10cls_e/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.5 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 100 --cls_epochs 10 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --lr-scheduler "step" --lr-steps "50-75" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_0.5_lr_step_50-75/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_0.5_lr_step_50-75/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # # Train using SupCon loss, with reweighting and large LR and BS
        # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=64G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed}_1k_bs --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 1. \
        #                 --batch_size 1024 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_large_lr_bs/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_large_lr_bs/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # # Train using SupCon loss, simple ERM (appendix)
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-supcon-erm-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --train_from_scratch --resnet_width ${width} --loss_fn supcon \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_erm/celebA_erm_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_erm/celebA_erm_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # # Train using SupCon loss, with reweighting, and a cosine learning rate decay
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=64G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed}-revised-bs256 --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 256 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 500 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_revised_bs256/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_revised_bs256/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # # Train using SupCon loss, with reweighting, and a cosine learning rate decay
        # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=64G \
        #             --kill-on-bad-exit --job-name celebA-supcon-reweight-w_${width}-s_${seed}-revised-bs128 --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 500 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_revised_bs128_new/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon_reweight_revised_bs128_new/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-center-reweight-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "center_loss" --lr-scheduler "none" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_reweight/celebA_reweight_width_${width}_seed_${seed}_center/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_reweight/celebA_reweight_width_${width}_seed_${seed}_center.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-center-reweight-w_${width}-s_${seed}-lambda-0.05-cosine --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.05 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "center_loss" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_reweight_lambda_0.05_cosine/celebA_reweight_width_${width}_seed_${seed}_center/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_reweight_lambda_0.05_cosine/celebA_reweight_width_${width}_seed_${seed}_center.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-center-reweight-w_${width}-s_${seed}-lambda-0.01-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "center_loss" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_reweight_lambda_0.01_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_center/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_reweight_lambda_0.01_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_center.log 2>&1 &

        # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-distill-reweight-w_${width}-s_${seed}-lambda-1-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "distillation" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_reweight_lambda_1_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_reweight_lambda_1_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-distill-center-reweight-w_${width}-s_${seed}-lambda-1-0.1c-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "distillation_center_loss" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_center_reweight_lambda_1_0.1c_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill_center/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_center_reweight_lambda_1_0.1c_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill_center.log 2>&1 &
        
        # TODO: Train with pseudo MixUp
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-mixup-reweight-w_${width}-s_${seed}-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "ce_mixup" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_mixup_reweight_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_ce_mixup/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_mixup_reweight_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_ce_mixup.log 2>&1 &

        # TODO: Train with complete MixUp
        srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
                    --kill-on-bad-exit --job-name celebA-mixup-complete-reweight-w_${width}-s_${seed}-cosine-aug --nice=0 \
                    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
                    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                    /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
                        --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "ce_mixup_complete" --lr-scheduler "cosine" \
                        --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_mixup_complete_reweight_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_ce_mixup_complete/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_mixup_complete_reweight_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_ce_mixup_complete.log 2>&1 &
        
        # TODO: Train with MixUP and center loss
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-center-mixup-reweight-w_${width}-s_${seed}-lambda-1-0.01c-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "center_loss_mixup" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_mixup_reweight_lambda_1_0.01c_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_center_mixup/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_mixup_reweight_lambda_1_0.01c_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_center_mixup.log 2>&1 &

        # # TODO: LATEST exp
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-distill-from-subsample-reweight-w_${width}-s_${seed}-lambda-1-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "distillation" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill/ \
        #                 --distillation_checkpoint /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample_cosine_augment/celebA_subsample_width_224_seed_1_ce/last_model.pth > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill.log 2>&1 &

        # TODO: Double distillation
        # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-double-distill-from-subsample-reweight-w_${width}-s_${seed}-lambda-1-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "distillation" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment_second/celebA_reweight_width_${width}_seed_${seed}_distill_second/ \
        #                 --distillation_checkpoint /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_distill/last_model.pth > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment_second/celebA_reweight_width_${width}_seed_${seed}_distill_second.log 2>&1 &

        # TODO: Third distillation round
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-third-distill-from-subsample-reweight-w_${width}-s_${seed}-lambda-1-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "distillation" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment_third/celebA_reweight_width_${width}_seed_${seed}_distill_third/ \
        #                 --distillation_checkpoint /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment_second/celebA_reweight_width_${width}_seed_${seed}_distill_second/last_model.pth > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_from_subsample_reweight_lambda_1_cosine_augment_third/celebA_reweight_width_${width}_seed_${seed}_distill_third.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-distill-reweight-w_${width}-s_${seed}-lambda-1 --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "distillation" --lr-scheduler "none" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_reweight_lambda_1/celebA_reweight_width_${width}_seed_${seed}_distill/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_distill_reweight_lambda_1/celebA_reweight_width_${width}_seed_${seed}_distill.log 2>&1 &

        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-reweight-w_${width}-s_${seed}-lambda-0.01-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn "ce" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_reweight_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_reweight_cosine_augment/celebA_reweight_width_${width}_seed_${seed}_ce.log 2>&1 &

        # New

        # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-erm-w_${width}-s_${seed}-lambda-0.01-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --train_from_scratch --resnet_width ${width} --loss_fn "ce" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_erm_cosine_augment/celebA_erm_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_erm_cosine_augment/celebA_erm_width_${width}_seed_${seed}_ce.log 2>&1 &
        
        # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-subsample-w_${width}-s_${seed}-lambda-0.01-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --train_from_scratch --resnet_width ${width} --loss_fn "ce" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample_cosine_augment/celebA_subsample_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample_cosine_augment/celebA_subsample_width_${width}_seed_${seed}_ce.log 2>&1 &

        # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-center-erm-w_${width}-s_${seed}-lambda-0.01-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --train_from_scratch --resnet_width ${width} --loss_fn "center_loss" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_erm_cosine_augment/celebA_erm_width_${width}_seed_${seed}_center/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_erm_cosine_augment/celebA_erm_width_${width}_seed_${seed}_center.log 2>&1 &
        
        # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-center-erm-w_${width}-s_${seed}-lambda-0.01-cosine-aug --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon_new.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --cls_epochs 10 --center-loss-lambda 0.01 --augment_data --save_step 10000 --save_last --log_every 50 --subsample_to_minority --train_from_scratch --resnet_width ${width} --loss_fn "center_loss" --lr-scheduler "cosine" \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_subsample_cosine_augment/celebA_subsample_width_${width}_seed_${seed}_center/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_center_subsample_cosine_augment/celebA_subsample_width_${width}_seed_${seed}_center.log 2>&1 &
    done
done
