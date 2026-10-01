#!/usr/bin/env bash
#SBATCH --job-name=proto_headpose
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rose.gurung@maine.edu
#SBATCH --output=./runs/%j_headpose_classification.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128gb
#SBATCH --time=10-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

srun -u /home/rgurung/.conda/envs/agepred/bin/python main_tree.py \
    --dataset=head_pose_dataset \
    --pose_data_dir=/home/rgurung/projects/maniskill/Gaussian-Regression_Tree/data/300W_LP \
    --aflw_data_dir=/home/rgurung/projects/maniskill/Gaussian-Regression_Tree/data/AFLW2000 \
    --net=resnet50 \
    --depth=7 \
    --num_features=256 \
    --epochs=100 \
    --batch_size=128 \
    --lr=5e-3 \
    --lr_block=1e-4 \
    --lr_net=1e-5 \
    --lr_delta=5e-3 \
    --weight_decay=1e-3 \
    --freeze_epochs=0 \
    --milestones=35,65,90 \
    --gamma=0.5 \
    --prototype_temp=0.2 \
    --target_temp=0.01 \
    --anneal_start_frac=0.3 \
    --anneal_end_frac=0.7 \
    --label_smoothing=0.5 \
    --W1=1 --H1=1 \
    --log_dir=./runs/headpose_classification \
    --num_workers=2 \
    --train_subset=0.2
    # --label_smoothing_end=0.01 \
    
