#!/bin/bash

#SBATCH --job-name=Mozaf                # Job name
#SBATCH --cpus-per-task=2               # CPU cores
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=64gb                      # Job memory request
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:l40:1                # Request one GPU
#SBATCH --output=%j_prototree_face.log  # Standard output and error log

echo "====================================================="
pwd; hostname; date
echo "====================================================="
echo "ProtoTree | resnet50 | depth 4 (16 leaves, 5-year age bins) | face_dataset"

module load apptainer

# Point these at wherever the ga-reg CSVs (train_set.csv/test_set.csv) and the
# UTKFace images actually live on this machine.
export FACE_CSV_DIR="/home/mozafarinia/Documents/Gaussian-Regression_Tree/csv_dataset"
export FACE_DATA_ROOT="/home/mozafarinia/Documents/data/age_prediction/utkcropped"

epochs=100
log_dir="./runs/prototree_face"
dataset="face_dataset"
lr=0.001
lr_block=0.001
lr_net=1e-5
num_features=256
depth=4
net="resnet50"
freeze_epochs=30
milestones=60,70,80,90,100
batch_size=64
W1=1
H1=1

apptainer run --nv "/home/mozafarinia/Documents/sif_file_vit.sif" \
python "main_tree.py" --epochs=$epochs --log_dir=$log_dir \
--dataset=$dataset --lr=$lr \
--lr_block=$lr_block --lr_net=$lr_net --num_features=$num_features --depth=$depth \
--net=$net --freeze_epochs=$freeze_epochs --milestones=$milestones \
--batch_size=$batch_size --W1=$W1 --H1=$H1

echo "====================================================="
date
