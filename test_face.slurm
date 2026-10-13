#!/bin/bash

#SBATCH --job-name=Mozaf         # Job name
#SBATCH --cpus-per-task=2               # Run on a 2 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1              # Request one GPU

#SBATCH --output=%j_face_test_mae.log   # Standard output and error log

echo "====================================================="
pwd; hostname; date
echo "====================================================="

module load apptainer

run_dir="./runs/prototree_face/run_N"      # finished training run to evaluate
checkpoint="pruned_and_projected"          # final saved model (or best_valid_model, pruned, last_epoch)
csv_dir="./protopairnet_csv_dataset"
export FACE_DATA_ROOT="./data/face_dataset/utkcropped"   # UTKFace images on the cluster
sif_file="/home/mozafarinia/Documents/car-regression/sif_file_car.sif"

apptainer run --nv "$sif_file" \
python "eval_test_mae.py" --run_dir=$run_dir --checkpoint=$checkpoint --csv_dir=$csv_dir

echo "====================================================="
date
