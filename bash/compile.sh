#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --mem=15GB
#SBATCH --ntasks=1
module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
            
python /hpc/home/csr33/ast_object_detection/src/compile_predictions.py --detect_tank_dir $detect_tank_dir --compiled_data_path $compiled_data_path
