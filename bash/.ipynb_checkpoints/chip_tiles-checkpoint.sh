#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --cpus-per-task=20
#SBATCH --mem=75GB
#SBATCH --ntasks=1
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo


python /hpc/home/csr33/ast_object_detection/src/chip_tiles.py --tile_dir $tile_dir --img_dir $img_dir --connections 20 --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID --tilename_chunks_path $tilename_chunks_path