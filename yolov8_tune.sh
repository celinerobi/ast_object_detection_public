#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --mem=30GB # Memory per node. 5 GB
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=10
#SBATCH --output /hpc/home/csr33/ast_object_detection/tune.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/tune.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8_env

#export COMET_API_KEY=CwpWEkJDRJc0rY57WYgoDuJvp

python /hpc/home/csr33/ast_object_detection/tune.py --data "/hpc/home/csr33/ast_object_detection/ast.yaml" --model "yolov8n.pt"
#yolo task=detect mode=train model=yolov8n.pt imgsz=640 data=ast.yaml epochs=50 batch=8 name=yolov8n_8b_50e
