#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --mem=100GB # Memory per node. 5 GB
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=20
#SBATCH --output /hpc/home/csr33/ast_object_detection/yolov8/bash/output/tune.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/yolov8/bash/error/tune.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8

#export COMET_API_KEY=CwpWEkJDRJc0rY57WYgoDuJvp
python /hpc/home/csr33/ast_object_detection/yolov8/src/tune.py --data "/hpc/home/csr33/ast_object_detection/ast.yaml" --model "yolov8n.pt" --workers 10 --iterations 100