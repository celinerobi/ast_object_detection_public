#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=50GB # Memory per node
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/naip_in_slosh.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/naip_in_slosh.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8


python /hpc/home/csr33/ast_object_detection/images_for_prediction/src/naip_in_slosh.py --num_chunks $num_chunks --chunked_naip_data_dir $chunked_naip_data_dir --chunked_naip_data_filename $chunked_naip_data_filename