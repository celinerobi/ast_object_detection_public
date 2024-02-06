#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --mem-per-cpu=100GB
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/predict_preprocessing.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/predict_preprocessing.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

python /hpc/home/csr33/ast_object_detection/src/preprocessing.py --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID
