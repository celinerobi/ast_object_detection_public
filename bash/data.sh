#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --mem=5GB # Memory per node. 5 GB
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/data.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/data.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
    
                 
python /hpc/home/csr33/ast_object_detection/src/data.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15