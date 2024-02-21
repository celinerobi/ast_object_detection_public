#!/bin/bash
#SBATCH --partition=scavenger-gpu
#SBATCH --mem=40GB 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/val.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/val.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8

#export COMET_API_KEY=CwpWEkJDRJc0rY57WYgoDuJvp

# Update settings
# https://docs.ultralytics.com/quickstart/
                 
# Update a setting
#yolo settings runs_dir="/work/csr33/object_detection/runs" weights_dir="/work/csr33/object_detection/weights"
#yolo detect val data="/hpc/home/csr33/ast_object_detection/ast.yaml" model="/work/csr33/object_detection/runs/detect/train_w_tuned_hyperparameters3/weights/best.pt" save_json=True split="train"
python /hpc/home/csr33/ast_object_detection/src/val.py --model "/work/csr33/object_detection/runs/detect/train_w_tuned_hyperparameters3/weights/best.pt"