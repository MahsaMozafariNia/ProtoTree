#!/bin/bash

#SBATCH --job-name=Mozaf         # Job name
#SBATCH --cpus-per-task=4               # Run on a 4 cores per node (one per DataLoader worker)
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a30:1              # Request one GPU

#SBATCH --output=%j_face_classification_train.log   # Standard output and error log

echo "====================================================="
pwd; hostname; date
git rev-parse --abbrev-ref HEAD; git rev-parse HEAD
echo "====================================================="

module load apptainer

# Original ProtoTree (exp(-distance) similarity) on face_dataset with the same age bins as run_5
export FACE_CSV_DIR="./protopairnet_csv_dataset"
export FACE_DATA_ROOT="/home/mozafarinia/Documents/vit_age_prediction-rose-setup/data/age_prediction/utkcropped"   # UTKFace images on the cluster
sif_file="/home/mozafarinia/Documents/car-regression/sif_file_car.sif"

epochs=100
log_dir="./runs/prototree_face_classification"   # main's ProtoTree does not version runs: a new job overwrites this folder
dataset="face_dataset"
lr=0.001
lr_block=0.001
lr_net=1e-5
num_features=128
depth=5
net="resnet50"
freeze_epochs=5
milestones=50,100,120,140,160,180
batch_size=128
weight_decay=5e-4
W1=1
H1=1
# bin_strategy="quantile"

apptainer run --nv "$sif_file" \
python "main_tree.py" --epochs=$epochs --log_dir=$log_dir \
--dataset=$dataset --lr=$lr \
--lr_block=$lr_block --lr_net=$lr_net --num_features=$num_features --depth=$depth \
--net=$net --freeze_epochs=$freeze_epochs --milestones=$milestones \
--batch_size=$batch_size --W1=$W1 --H1=$H1 --weight_decay=$weight_decay --pretrained_face --num_workers=4 \
# --double_leaves --bin_strategy=$bin_strategy

echo "====================================================="
date
