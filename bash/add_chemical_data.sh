#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --mem=15GB
#SBATCH --ntasks=1
module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
            
python /hpc/home/csr33/ast_object_detection/src/add_chemical_data.py --compile_data_path $compile_data_path --tri_with_sg_path $tri_with_sg_path 