from ultralytics import YOLO
import argparse
import os
from ultralytics import settings
#import torch
#torch.cuda.set_device(0) # Set to your desired GPU number


def get_args_parse():
    parser = argparse.ArgumentParser("Yolv8 model train with option for hyperparameter setting")
    parser.add_argument("--model", default='yolov8x.pt', type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--workers", default=8, type=int)

    parser.add_argument("--name", type=str)
    parser.add_argument("--project_name", default="/work/csr33/object_detection", type=str)
    parser.add_argument("--tune_yaml", type=str)
    parser.add_argument("--data_yaml", default="/hpc/home/csr33/ast_object_detection/ast.yaml", type=str)

    args = parser.parse_args()
    return args

def train(args):
    os.chdir("/work/csr33/object_detection")
    # Update settings
    # https://docs.ultralytics.com/quickstart/
    os.makedirs('/work/csr33/object_detection/weights', exist_ok=True)
    settings.update({'weights_dir': '/work/csr33/object_detection/weights'})
    os.makedirs('/work/csr33/object_detection/runs', exist_ok=True)
    settings.update({'runs_dir': '/work/csr33/object_detection/runs'})

    settings.update({'neptune': False, 'clearml': False, 
                     "raytune": False, 'comet': False, 
                     'dvc': False, 'hub': False, 'mlflow': False,
                     'tensorboard': False, 'wandb': True})
    

    if args.name == None:
        args.name = f'{os.path.splitext(args.model)[0]}_e{args.epochs}_imgsz{args.imgsz}'
    # Load the model.
    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)
    # Training.
    results = model.train(data = args.data_yaml, name = args.name, imgsz=args.imgsz, 
                          cfg=args.tune_yaml, workers=args.workers)
    
    
if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    train(args)


