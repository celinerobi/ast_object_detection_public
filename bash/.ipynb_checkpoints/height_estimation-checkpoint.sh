#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/height_estimation.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/height_estimation.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8


python /hpc/home/csr33/ast_object_detection/src/height_estimation.py --collection "3dep-lidar-hag" --ast_data_path "/hpc/group/borsuklab/ast/tile_level_annotation_multiple_capture_date_neighbor_tile_removed/tile_level_annotation_multiple_capture_date_neighbor_tile_removed.geojson"


