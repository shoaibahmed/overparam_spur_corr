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

for seed in 0 1 2; do
    for width in 1 2 4 6 8 16 32 48 64 80 88 96; do
        # Train using CE loss
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-ce-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn ce \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce/celebA_reweight_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce/celebA_reweight_width_${width}_seed_${seed}_ce.log 2>&1 &

        # Train using CE loss, and subsampling
        srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
                    --kill-on-bad-exit --job-name celebA-ce-subsample-w_${width}-s_${seed} --nice=0 \
                    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
                    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                    /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
                        --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 500 --save_step 10000 --save_last --log_every 50 --subsample_to_minority --train_from_scratch --resnet_width ${width} --loss_fn ce \
                        --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample/celebA_subsample_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_subsample/celebA_subsample_width_${width}_seed_${seed}_ce.log 2>&1 &
        
        # Train using CE loss, simple ERM (appendix)
        srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
                    --kill-on-bad-exit --job-name celebA-ce-erm-w_${width}-s_${seed} --nice=0 \
                    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
                    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                    /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
                        --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --train_from_scratch --resnet_width ${width} --loss_fn ce \
                        --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_erm/celebA_erm_width_${width}_seed_${seed}_ce/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_ce_erm/celebA_erm_width_${width}_seed_${seed}_ce.log 2>&1 &

        # Train using SupCon loss
        # srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        #             --kill-on-bad-exit --job-name celebA-supcon-w_${width}-s_${seed} --nice=0 \
        #             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        #             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        #             /opt/conda/bin/python /netscratch/siddiqui/Repositories/overparam_spur_corr/run_expt_supcon.py -s confounder -d CelebA -t Blond_Hair -c Male --fraction 1.0 --lr 0.01 \
        #                 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --seed ${seed} --n_epochs 50 --save_step 10000 --save_last --log_every 50 --reweight_groups --train_from_scratch --resnet_width ${width} --loss_fn supcon \
        #                 --log_dir /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon/celebA_reweight_width_${width}_seed_${seed}_supcon/ > /netscratch/siddiqui/Repositories/overparam_spur_corr/output_supcon/celebA_reweight_width_${width}_seed_${seed}_supcon.log 2>&1 &
    done
done
