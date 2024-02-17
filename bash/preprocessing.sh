#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/predict_preprocessing.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/predict_preprocessing.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

python /hpc/home/csr33/ast_object_detection/images_for_prediction/src/preprocessing.py --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID --processing_naip_dir $processing_naip_dir --processing_naip_filename $processing_naip_filename

