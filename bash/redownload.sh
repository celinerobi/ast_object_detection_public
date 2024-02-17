#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/redownload.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/redownload.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8


python /hpc/home/csr33/ast_object_detection/src/redownload.py --tile_dir "/work/csr33/images_for_predictions/naip_tiles" --naip_data_path "/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet" --connections 20