#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100GB
#SBATCH --ntasks=1
#SBATCH --output //hpc/home/csr33/ast_object_detection/predict_preprocessing.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/predict_preprocessing.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

python /hpc/home/csr33/ast_object_detection/predict_preprocessing.py --processing_naip_df "/work/csr33/images_for_predictions/processed_naip_data.parquet" --naip_tile_df "/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet" --imgsz 640
