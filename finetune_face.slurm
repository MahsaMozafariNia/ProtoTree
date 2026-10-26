#!/bin/bash

#SBATCH --job-name=Mozaf         # Job name
#SBATCH --cpus-per-task=2               # Run on a 2 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=04:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1              # Request one GPU

#SBATCH --output=%j_face_finetune.log   # Standard output and error log

echo "====================================================="
pwd; hostname; date
git rev-parse --abbrev-ref HEAD; git rev-parse HEAD
echo "====================================================="

module load apptainer

run_dir="./runs/prototree_face_with_protopairnet_data/run_5"   # finished training run to fine-tune
export FACE_CSV_DIR="/home/mozafarinia/Documents/ProtoTree/protopairnet_csv_dataset"
export FACE_DATA_ROOT="/home/mozafarinia/Documents/vit_age_prediction-rose-setup/data/age_prediction/utkcropped"   # UTKFace images on the cluster
sif_file="/home/mozafarinia/Documents/car-regression/sif_file_car.sif"

apptainer run --nv "$sif_file" \
python "finetune_projected.py" --run_dir=$run_dir --epochs=20 --anneal_epochs=10 --temp_start=0.05 --lr_margin=1e-3

echo "====================================================="
date
