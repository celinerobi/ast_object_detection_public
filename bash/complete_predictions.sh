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
output_dir="/hpc/home/csr33/ast_object_detection/bash/output"
error_dir="/hpc/home/csr33/ast_object_detection/bash/error"

height_estimation_dir="/hpc/group/borsuklab/csr33/object_detection/height_estimation"
prediction_dir="/hpc/group/borsuklab/csr33/object_detection/predictions"
img_dir="/hpc/group/borsuklab/csr33/object_detection/imgs"
tile_dir="/hpc/group/borsuklab/csr33/object_detection/naip_tiles"
tilename_chunks_path="/hpc/home/csr33/ast_object_detection/tilename_chunks.npz"
collection="3dep-lidar-hag"

backoff_factor=10
max_retries=10

compiled_data_path="/hpc/group/borsuklab/csr33/object_detection/compiled_predicted_tank.geojson"



python /hpc/home/csr33/ast_object_detection/src/tilename_chunk.py --num_chunks $num_chunks --tile_dir $tile_dir --tilename_chunks_path $tilename_chunks_path
#sleep 5

#rm -rf $prediction_dir/*
predict_jobname="predict"
predict_script="/hpc/home/csr33/ast_object_detection/bash/predict.sh"
predict_job_id=$(sbatch  --array=0-$n_arrays --job-name $predict_jobname --output $output_dir/$predict_jobname/$predict_jobname"_%a.out" --error $error_dir/$predict_jobname/$predict_jobname"_%a.err" --export=imgsz=$imgsz,model_path=$model_path,prediction_dir=$prediction_dir,tile_dir=$tile_dir,tilename_chunks_path=$tilename_chunks_path,img_dir=$img_dir $predict_script | awk '{print $4}')

height_estimation_jobname="height_estimation"
height_estimation_script="/hpc/home/csr33/ast_object_detection/bash/height_estimation.sh"
height_estimation_job_id=$(sbatch --array=270,930,877,628 --job-name $height_estimation_jobname --output $output_dir/$height_estimation_jobname/$height_estimation_jobname"_%a.out" --error $error_dir/$height_estimation_jobname/$height_estimation_jobname"_%a.err" --export=prediction_dir=$prediction_dir,height_estimation_dir=$height_estimation_dir,collection=$collection,backoff_factor=$backoff_factor,max_retries=$max_retries $height_estimation_script | awk '{print $4}')
#--dependency=aftercorr:$predict_job_id 


compile_script="/hpc/home/csr33/ast_object_detection/bash/compile.sh"
compile_jobname="compile"
compile_job_id=$(sbatch --job-name $compile_jobname --output $output_dir/$compile_jobname"_%a.out" --error $error_dir/$compile_jobname"_%a.err" --export=compiled_data_path=$compiled_data_path,detect_tank_dir=$height_estimation_dir $compile_script | awk '{print $4}')