#!/bin/bash
#SBATCH --partition scavenger-gpu
#SBATCH --mem=25GB
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output /hpc/home/csr33/ast_object_detection/bash/output/predict_preprocessing.out
#SBATCH --error /hpc/home/csr33/ast_object_detection/bash/error/bapredict_preprocessing.err
echo "load envirionment"
module unload Anaconda3/2021.05
CONDA_BASE=/hpc/home/csr33/miniconda3
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /hpc/group/borsuklab/cred/.conda/envs/yolov8
echo
python /hpc/home/csr33/ast_object_detection/images_for_prediction/src/predict.py --imgsz $imgsz --chunk_id $SLURM_ARRAY_TASK_ID --model_path $model_path --prediction_dir $prediction_dir --prediction_filename $prediction_filename


    parser.add_argument("--tile_dir", default="/work/csr33/images_for_predictions/naip_tiles", type=str)
    parser.add_argument("--tilename_chunks_path", default='/hpc/home/csr33/ast_object_detection/images_for_prediction/tilename_chunks.npz', type=str)
    parser.add_argument("--model_path", default="/work/csr33/object_detection/runs/detect/baseline_train/weights/best.pt", type=str)
    parser.add_argument("--prediction_dir", default="/work/csr33/images_for_predictions/predictions", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument('--img_dir', type=str, default="/work/csr33/images_for_predictions/naip_imgs")
