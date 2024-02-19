#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --mem=40GB # Memory per node. 5 GB
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=20
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/train_w_tuned_hyperparameters.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/train_w_tuned_hyperparameters.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
                 
/hpc/home/csr33/ast_object_detection/src/train.py --model "yolov8x.pt" --epochs 300 --workers 9 --name "train_w_tuned_hyperparameters" --imgsz 640 --tune_yaml "/work/csr33/object_detection/runs/detect/tune10/best_hyperparameters.yaml" --data "/hpc/home/csr33/ast_object_detection/ast.yaml"

    
