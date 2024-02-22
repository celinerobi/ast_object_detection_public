import wandb
import argparse
import os

from ultralytics import YOLO
from ultralytics import settings

#https://docs.ultralytics.com/reference/utils/tuner/
#https://docs.ultralytics.com/reference/utils/tuner/#ultralytics.utils.tuner.run_ray_tune
def get_args_parse():
    parser = argparse.ArgumentParser("Tune yolov8 using genetic algorithm")
    parser.add_argument("--data", default="/hpc/home/csr33/ast_object_detection/ast.yaml", type=str)
    parser.add_argument("--model", default='yolov8n.pt', type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--iterations", default=300, type=int)
    parser.add_argument("--workers", default=8, type=int)

    args = parser.parse_args()
    return args

def tune(args):
    os.chdir("/work/csr33/object_detection")
    # Update settings
    # https://docs.ultralytics.com/quickstart/
    os.makedirs('/work/csr33/object_detection/weights', exist_ok=True)
    settings.update({'weights_dir': '/work/csr33/object_detection/weights'})

    os.makedirs('/work/csr33/object_detection/runs', exist_ok=True)
    settings.update({'runs_dir': '/work/csr33/object_detection/runs'})

    settings.update({'neptune': False, 'clearml': False, 
                     "raytune": True, #'comet': True, 'raytune': True,
                     'dvc': False, 'hub': False, 'mlflow': False,
                     'tensorboard': False, 'wandb': True})

    run = wandb.init(project='yolov8-ast-tune')
    # Initialize the YOLO model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)
    # Tune hyperparameters on COCO8 for 30 epochs
    # Use model.tune to tune the hyperparameters
    best_params = model.tune(data=args.data, epochs=args.epochs, 
                             iterations=args.iterations, workers=args.workers)
    # Log the best hyperparameters to WandB
    wandb.config.update(best_params)
    # Log the model to WandB
    wandb.save(f"yolov8_e{args.epochs}_i{args.iterations}.pth")
    
    
if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    tune(args)
