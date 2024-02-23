#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8


num_chunks=1000
n_arrays=$(( $num_chunks - 1 ))
imgsz=640
model_path="/work/csr33/object_detection/runs/detect/train_w_tuned_hyperparameters3/weights/best.pt"
prediction_dir="/work/csr33/images_for_predictions/predictions"
img_dir="/work/csr33/images_for_predictions/naip_imgs"
output_dir="/hpc/home/csr33/ast_object_detection/bash/output"
error_dir="/hpc/home/csr33/ast_object_detection/bash/error"
tile_dir="/work/csr33/images_for_predictions/naip_tiles"
tilename_chunks_path="/hpc/home/csr33/ast_object_detection/tilename_chunks.npz"

            
tri_with_sg_path="/hpc/home/csr33/tri_with_specific_gravity.geojson"
SLURM_ARRAY_TASK_ID=0
python /hpc/home/csr33/ast_object_detection/src/add_chemical_data.py --chunk_id $SLURM_ARRAY_TASK_ID --prediction_dir $prediction_dir --tri_with_sg_path $tri_with_sg_path