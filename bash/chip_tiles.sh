#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/output/chip_tiles.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/error/chip_tiles.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

python /hpc/home/csr33/ast_object_detection/images_for_prediction/src/chip_tiles.py --tile_dir "/work/csr33/images_for_predictions/naip_tiles" --img_dir "/work/csr33/images_for_predictions/naip_imgs" --connections 16 --imgsz 640 --chunk_id 3 --tilename_chunks_path "/hpc/home/csr33/ast_object_detection/images_for_prediction/tilename_chunks.npz"


