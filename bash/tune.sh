#!/bin/bash
#SBATCH --partition=gpu-common
#SBATCH --mem=100GB # Memory per node. 5 GB
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 # Number of GPUs per node
#SBATCH --cpus-per-task=10
#SBATCH --output /work/csr33/bash_outputs/output/tune_200_iter_20_epochs_20_cpu_AdamW_optimizer_n_model.out
#SBATCH --error /work/csr33/bash_outputs/error/tune_200_iter_20_epochs_20_cpu_AdamW_optimizer_n_model.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8

#export COMET_API_KEY=CwpWEkJDRJc0rY57WYgoDuJvp
python /hpc/home/csr33/ast_object_detection/src/tune.py --data "/hpc/home/csr33/ast_object_detection/ast.yaml" --model "yolov8n.pt" --workers 8 --iterations 100 --epochs 20 --optimizer "AdamW"