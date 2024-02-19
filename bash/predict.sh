#!/bin/bash
#SBATCH --partition gpu-common
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/predict.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/predict.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo

num_chunks=750
n_arrays=$(( $num_chunks - 1 ))
imgsz=640
model_path="/work/csr33/object_detection/runs/detect/baseline_train/weights/best.pt"
prediction_dir="/work/csr33/images_for_predictions/predictions"
img_dir="/work/csr33/images_for_predictions/naip_imgs"
output_dir="/hpc/home/csr33/ast_object_detection/bash/output"
error_dir="/hpc/home/csr33/ast_object_detection/bash/error"
SLURM_ARRAY_TASK_ID=0
tile_dir="/work/csr33/images_for_predictions/naip_tiles"
tilename_chunks_path="/hpc/home/csr33/ast_object_detection/tilename_chunks.npz"


python /hpc/home/csr33/ast_object_detection/src/predict.py --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID --model_path $model_path --prediction_dir $prediction_dir --tile_dir $tile_dir --tilename_chunks_path $tilename_chunks_path --img_dir $img_dir