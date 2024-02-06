#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/predict_preprocessing.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/bapredict_preprocessing.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

python /hpc/home/csr33/ast_object_detection/src/predict.py --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID --model_path $model_path
