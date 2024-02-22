#!/bin/bash
#SBATCH --partition scavenger
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/process_tri_data.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/process_tri_data.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo
tri_with_sg_path="/hpc/home/csr33/tri_with_specific_gravity.csv"
detected_tanks_path="/work/csr33/images_for_predictions/predictions/merged_predictions_0.csv"
naics_industry_codes_path="/hpc/home/csr33/spatial-match-ast-chemicals/naics_industry_keys.csv"
tri_2022_us_path="/hpc/group/borsuklab/csr33/chemical_data/tri/2022_us.csv"

python /hpc/home/csr33/ast_object_detection/src/process_tri_data.py --tri_with_sg_path $tri_with_sg_path --naics_industry_codes_path $naics_industry_codes_path --tri_2022_us_path $tri_2022_us_path