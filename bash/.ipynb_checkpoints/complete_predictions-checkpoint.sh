#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=50M # Memory per node
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/complete.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/complete.err

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

python /hpc/home/csr33/ast_object_detection/src/tilename_chunk.py --num_chunks $num_chunks --tile_dir $tile_dir --tilename_chunks_path $tilename_chunks_path
sleep 5

rm -rf $prediction_dir/*
predict_jobname="predict"
predict_script="/hpc/home/csr33/ast_object_detection/bash/predict.sh"
predict_job_id=$(sbatch  --array=0-$n_arrays --job-name $predict_jobname --output $output_dir/$predict_jobname"_%a.out" --error $error_dir/$predict_jobname"_%a.err" --export=imgsz=$imgsz,model_path=$model_path,prediction_dir=$prediction_dir,tile_dir=$tile_dir,tilename_chunks_path=$tilename_chunks_path,img_dir=$img_dir $predict_script | awk '{print $4}')


