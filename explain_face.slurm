#!/bin/bash

#SBATCH --job-name=Mozaf         # Job name
#SBATCH --cpus-per-task=2               # Run on a 2 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=gpu                 # Select partition
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:a30:1              # Request one GPU

#SBATCH --output=%j_face_explain_local.log   # Standard output and error log

echo "====================================================="
pwd; hostname; date
git rev-parse --abbrev-ref HEAD; git rev-parse HEAD
echo "====================================================="

module load apptainer

run_dir="./runs/prototree_face_with_protopairnet_data/run_5"   # finished training run to explain
checkpoint="pruned_and_projected"          # subfolder of <run_dir>/checkpoints, or a full path (e.g. $run_dir/finetune/run_1/checkpoints/finetuned_best)
csv_dir="./protopairnet_csv_dataset"
num_test_samples=100                        # explain the first N images of test_set.csv
export FACE_DATA_ROOT="/home/mozafarinia/Documents/vit_age_prediction-rose-setup/data/age_prediction/utkcropped"   # UTKFace images on the cluster
sif_file="/home/mozafarinia/Documents/car-regression/sif_file_car.sif"

# predvis.pdf is rendered with Graphviz; check it exists inside the container
apptainer exec "$sif_file" which dot || echo "WARNING: Graphviz 'dot' not found in container, predvis.pdf will fail"

apptainer run --nv "$sif_file" \
python "main_explain_local.py" --run_dir=$run_dir --checkpoint=$checkpoint --csv_dir=$csv_dir \
--num_test_samples=$num_test_samples

echo "Results in $run_dir/local_explanations"
echo "====================================================="
date
