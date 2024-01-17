from ultralytics import YOLO
import comet_ml
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser("Exploratory Data Analysis")
    parser.add_argument("--data", default="/hpc/home/csr33/ast_object_detection/ast.yaml", type=str)
    parser.add_argument("--model", default='yolov8n.pt', type=str)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    return args

import wandb
from yolov8_module import YOLOv8  # Import your YOLOv8 module

# Define your YOLOv8 training function
def train(config):
    # Configure YOLOv8 hyperparameters
        model = YOLO(args.model)  # load a pretrained model (recommended for training)

    model = YOLOv8(
        num_classes=config.num_classes,
        backbone=config.backbone,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        # Add other YOLOv8-specific hyperparameters
    )
    
    # Your training logic here using the configured YOLOv8 model
    # ...

    # Log hyperparameters
    wandb.config.update({
        "num_classes": config.num_classes,
        "backbone": config.backbone,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        # Add other YOLOv8-specific hyperparameters
    })

    # Log metrics (replace with actual metrics from your training)
    wandb.log({"accuracy": 0.85, "loss": 0.12})


# Define hyperparameter sweep configuration for YOLOv8
sweep_config = {
    "method": "random",
    "parameters": {
        "num_classes": {"values": [1, 2, 3]},  # Example values, adjust as needed
        "backbone": {"values": ["resnet50", "darknet53"]},  # Example values
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"min": 4, "max": 32},
        # Add other YOLOv8-specific hyperparameters and ranges
    },
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="your-project-name")

# Define and run the agent
def run():
    # Run the agent with your YOLOv8 training function
    wandb.agent(sweep_id, function=train)

# Run the YOLOv8 hyperparameter optimization
run()


def train(args):
    comet_ml.init(project_name="yolov8-ast")

    # Load the model.
    # Load a model

    # Training.
    results = model.train(
       data=args.data,
       imgsz=args.imgsz,
       epochs=args.epochs,
       batch=args.batch,
       name=f'{os.path.splitext(args.model)[0]}_e{args.epochs}_b{args.batch}_imgsz{args.imgsz}')

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    train(args)
