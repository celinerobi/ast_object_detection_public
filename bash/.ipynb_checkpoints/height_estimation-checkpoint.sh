#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8

python /hpc/home/csr33/ast_object_detection/src/height_estimation.py --collection $collection --chunk_id $SLURM_ARRAY_TASK_ID --prediction_dir $prediction_dir
