import wandb
import argparse
import os
import numpy as np

from ultralytics import YOLO
from ultralytics import settings

def get_args_parse():
    parser = argparse.ArgumentParser("Valdate the model")
    parser.add_argument("--model_path", type=str, help="path to training weights for trained model")
    args = parser.parse_args()
    return args

def val(args):
    os.chdir("/work/csr33/object_detection")
    os.makedirs('/work/csr33/object_detection/weights', exist_ok=True)
    settings.update({'weights_dir': '/work/csr33/object_detection/weights'})
    os.makedirs('/work/csr33/object_detection/runs', exist_ok=True)
    settings.update({'runs_dir': '/work/csr33/object_detection/runs'})

    # Load a model
    model = YOLO(args.model_path)  # load a pretrained model (recommended for training)    
     
    print("val")
    # Validate the model
    metrics = model.val(split="val")  # no arguments needed, dataset and settings remembered
    # a list contains map50-95 of each category
    print("val map50-95 overall", metrics.box.map)
    print("val f1 score", metrics.box.f1)
    accuracy = np.trace(metrics.confusion_matrix.matrix) / np.sum(metrics.confusion_matrix.matrix)
    print("val accuracy", accuracy)  
    
    print("test")
    # Validate the model
    metrics = model.val(split="test")  # no arguments needed, dataset and settings remembered
    # a list contains map50-95 of each category
    print("test map50-95 overall", metrics.box.map)
    print("test f1 score", metrics.box.f1)
    accuracy = np.trace(metrics.confusion_matrix.matrix) / np.sum(metrics.confusion_matrix.matrix)
    print("test accuracy", accuracy)
    

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    val(args)
