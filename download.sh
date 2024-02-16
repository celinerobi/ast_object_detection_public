#!/bin/bash
#SBATCH --partition scavenger
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

python /hpc/home/csr33/ast_object_detection/images_for_prediction/src/download.py --connections $connections --tile_dir $tile_dir --chunk_id $chunk_id
