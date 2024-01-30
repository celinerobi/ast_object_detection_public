from ultralytics import YOLO
import comet_ml
from comet_ml import Experiment
import argparse
#https://docs.ultralytics.com/reference/utils/tuner/
#https://docs.ultralytics.com/reference/utils/tuner/#ultralytics.utils.tuner.run_ray_tune
def get_args_parse():
    parser = argparse.ArgumentParser("Exploratory Data Analysis")
    parser.add_argument("--data", default="/hpc/home/csr33/ast_object_detection/ast.yaml", type=str)
    parser.add_argument("--model", default='yolov8n.pt', type=str)
    args = parser.parse_args()
    return args

def tune(args):
    experiment = Experiment(
    api_key='CwpWEkJDRJc0rY57WYgoDuJvp',
    project_name='yolov8-ast-tume',)
    # Load the model.
    # Initialize the YOLO model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)
    # Tune hyperparameters on COCO8 for 30 epochs
    # Use model.tune to tune the hyperparameters
    best_params = model.tune(data=args.data, epochs=30, iterations=10,
               plots=False, save=False, val=False)
    # Log the best hyperparameters to Comet ML
    experiment.log_parameters(best_params)
    # Train the model with the best hyperparameters
    model.train()
    # Log the model to Comet ML
    experiment.log_model("YOLOv8", "yolov8.pth")
    
    
    
if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    tune(args)
