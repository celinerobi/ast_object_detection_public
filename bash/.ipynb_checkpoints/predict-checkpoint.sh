#!/bin/bash
#SBATCH --partition scavenger-gpu
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1  
#SBATCH --ntasks=1
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo


python /hpc/home/csr33/ast_object_detection/src/predict.py --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID --model_path $model_path --prediction_dir $prediction_dir --tile_dir $tile_dir --tilename_chunks_path $tilename_chunks_path --img_dir $img_dir