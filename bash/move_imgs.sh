#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --mem=10GB
#SBATCH --ntasks=1
module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
            
python /hpc/home/csr33/ast_object_detection/src/move_imgs.py --chunk_id $SLURM_ARRAY_TASK_ID --tilename_chunks_path "/hpc/home/csr33/ast_object_detection/move_imgs.npz"