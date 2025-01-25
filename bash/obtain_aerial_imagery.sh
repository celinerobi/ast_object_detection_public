#!/bin/bash

#SBATCH --partition=scavenger
#SBATCH --mem=50M # Memory per node
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/complete.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/complete.err

module unload Anaconda3/2021.05
source /hpc/home/csr33/miniconda3/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8


chunked_naip_data_dir="/work/csr33/images_for_predictions/chunked_naip_data"
chunked_naip_data_filename="chunked_naip_data"
rm -rf $chunked_naip_data_dir/*
identify_naip_script="/hpc/home/csr33/ast_object_detection/bash/naip_in_slosh.sh"
identify_naip_jobname="identify_naip"
identify_naip_job_id=$(sbatch --job-name $identify_naip_jobname --output $output_dir/$identify_naip_jobname".out" --error  $error_dir/$identify_naip_jobname".err" --export=num_chunks=$num_chunks,chunked_naip_data_dir=$chunked_naip_data_dir,chunked_naip_data_filename=$chunked_naip_data_filename $identify_naip_script | awk '{print $4}')


ntasks=1
cpus_per_task=13
mem_per_cpu="20GB"
download_jobname="download"
download_script="/hpc/home/csr33/ast_object_detection/bash/download.sh"
for i in $(seq 0 $n_arrays); do
    echo "Current number: $i"
    sbatch  --cpus-per-task=$cpus_per_task --mem-per-cpu=$mem_per_cpu --ntasks=$ntasks --job-name $download_jobname --output $output_dir/$download_jobname$i".out" --error $error_dir/$download_jobname$i".err" --export=connections=$cpus_per_task,tile_dir=$tile_dir,chunk_id=$i $download_script 
    sleep 610
done


chip_jobname="chip_tiles"
chip_script="/hpc/home/csr33/ast_object_detection/bash/chip_tiles.sh"
predict_job_id=$(sbatch  --array=0-$n_arrays --job-name $chip_jobname --output $output_dir/$chip_jobname/$chip_jobname"_%a.out" --error $error_dir/$chip_jobname/$chip_jobname"_%a.err" --export=imgsz=$imgsz,tile_dir=$tile_dir,img_dir=$img_dir,tilename_chunks_path=$tilename_chunks_path $chip_script | awk '{print $4}')