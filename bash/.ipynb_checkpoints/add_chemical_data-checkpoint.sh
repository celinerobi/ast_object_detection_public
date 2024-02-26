#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --mem=15GB
#SBATCH --ntasks=1
module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
            
python /hpc/home/csr33/ast_object_detection/src/add_chemical_data.py --chunk_id $SLURM_ARRAY_TASK_ID --height_estimation_dir $height_estimation_dir --complete_predicted_data_dir $complete_predicted_data_dir --tri_with_sg_path $tri_with_sg_path 