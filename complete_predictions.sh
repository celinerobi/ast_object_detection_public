#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=50M # Memory per node
#SBATCH --output /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/output/complete.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/error/complete.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8


num_chunks=250
n_arrays=$(( $num_chunks - 1 ))
imgsz=640
output_dir="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/output"
error_dir="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/error"

chunked_naip_data_dir="/work/csr33/images_for_predictions/chunked_naip_data"
chunked_naip_data_filename="chunked_naip_data"
rm -rf $chunked_naip_data_dir/*
identify_naip_script="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/naip_in_slosh.sh"
identify_naip_jobname="identify_naip"
identify_naip_job_id=$(sbatch --job-name $identify_naip_jobname --output $output_dir/$identify_naip_jobname".out" --error  $error_dir/$identify_naip_jobname".err" --export=num_chunks=$num_chunks,chunked_naip_data_dir=$chunked_naip_data_dir,chunked_naip_data_filename=$chunked_naip_data_filename $identify_naip_script | awk '{print $4}')


ntasks=1
cpus_per_task=13
mem_per_cpu="20GB"

tile_dir="/work/csr33/images_for_predictions/naip_tiles"
download_jobname="download"
download_script="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/download.sh"

for i in $(seq 0 $n_arrays); do
    echo "Current number: $i"
    sbatch  --cpus-per-task=$cpus_per_task --mem-per-cpu=$mem_per_cpu --ntasks=$ntasks --job-name $download_jobname --output $output_dir/$download_jobname$i".out" --error $error_dir/$download_jobname$i".err" --export=connections=$cpus_per_task,tile_dir=$tile_dir,chunk_id=$i $download_script 
    sleep 610
done
#--dependency=afterok:$identify_naip_job_id

#
#--dependency=afterok:$identify_naip_job_id

model_path="/work/csr33/object_detection/runs/detect/baseline_train/weights/best.pt"
prediction_dir="/work/csr33/images_for_predictions/predictions"
prediction_filename="predictions"
rm -rf $prediction_dir/*
predict_jobname="predict"
predict_script="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/predict.sh"
predict_job_id=$(sbatch  --array=0-$n_arrays --dependency=aftercorr:$preprocessing_job_id --job-name $predict_jobname --output $output_dir/$predict_jobname"_%a.out" --error $error_dir/$predict_jobname"_%a.err" --export=imgsz=$imgsz,model_path=$model_path,prediction_dir=$prediction_dir,prediction_filename=$prediction_filename $predict_script | awk '{print $4}')