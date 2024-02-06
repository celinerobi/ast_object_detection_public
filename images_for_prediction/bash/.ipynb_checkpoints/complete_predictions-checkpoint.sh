#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=50M # Memory per node
#SBATCH --output /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/output/complete.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/error/complete.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8



fragility_dir="/work/csr33/fragility"
test_size=0.2
lse_data_path="/work/csr33/fragility/lse_failure_sims_compiled.parquet"
failure_modes=("flotation" "sliding" "buckling")

num_chunks=100
n_arrays=$(( $num_chunks - 1 ))
imgsz=640

#SBATCH --output 
#SBATCH --error /hpc/home/csr33/ast_object_detection/images_for_prediction/bash/error/naip_in_slosh.err
output_dir="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/output"
error_dir="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/error"

identify_naip_script="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/naip_in_slosh.sh"
identify_naip_jobname="identify_naip"
identify_naip_job_id=$(sbatch --job-name $identify_naip_jobname --output $output_dir/$identify_naip_jobname".out" --error  $error_dir/$identify_naip_jobname".err" --export=num_chunks=$num_chunks $identify_naip_script | awk '{print $4}')


preprocessing_jobname="preprocessing"
preprocessing_script="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/preprocessing.sh"

preprocessing_job_id=$(sbatch  --array=0-$n_arrays --dependency=afterok:$identify_naip_job_id  --job-name $preprocessing_jobname  --output $output_dir/$preprocessing_jobname"_a.out" --error $error_dir/$preprocessing_jobname"_a.err" --export=imgsz=$imgsz $preprocessing_script | awk '{print $4}') 


predict_jobname="predict"
predict_script="/hpc/home/csr33/ast_object_detection/images_for_prediction/bash/predict.sh"
model_path="/work/csr33/object_detection/runs/detect/baseline_train/weights/best.pt"
preprocessing_job_id=$(sbatch  --array=0-$n_arrays --dependency=aftercorr:$preprocessing_job_id  --job-name $predict_jobname --output $output_dir/$predict_jobname"_a.out" --error $error_dir/$predict_jobname"_a.err" --export=imgsz=$imgsz,model_path=$model_path $predict_script | awk '{print $4}')