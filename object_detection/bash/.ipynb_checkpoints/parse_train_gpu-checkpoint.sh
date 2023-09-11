#!/bin/bash
#SBATCH --job-name=gpu_parse_model_train.job
#SBATCH --output=/hpc/home/csr33/ast_object_detection/object_detection/output/bash/parse_model_output.out
#SBATCH --error=/hpc/home/csr33/ast_object_detection/object_detection/error/bash/parse_model.err
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail #send email when job begins
#SBATCH --mail-user=csr33@duke.edu
#SBATCH --partition=gpu-common
#SBATCH --mem=10G 
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1     # Request 1 GPU

CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
echo "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /hpc/group/borsuklab/cred/frcnn

python /hpc/home/csr33/ast_object_detection/object_detection/parse.py --data_directory /hpc/group/borsuklab/ast/complete-dataset --parent_directory /hpc/group/borsuklab/faster-rcnn --img_directory chips_positive --annotation_directory chips_positive_corrected_xml --path_to_predefined_classes /hpc/home/csr33/ast_object_detection/object_detection/predefined_classes.txt

#CUDA_LAUNCH_BLOCKING=1 python /hpc/home/csr33/cred/AST_dataset/object_detection/model_train.py --parent_directory /hpc/group/borsuklab/cred/ast_complete_dataset --path_to_predefined_classes /hpc/home/csr33/cred/AST_dataset/object_detection/predefined_classes.txt --num_workers 1 --val_size 0.2 --scheduler_name exponentiallr --lr 0.010 --lr_gamma 0.9 --scheduler_name exponentiallr --batch_size 2 --num_workers 1 --num_epochs 50


