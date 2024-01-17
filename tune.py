from ultralytics import YOLO
#import comet_ml
import wandb
import argparse
#https://docs.ultralytics.com/reference/utils/tuner/
#https://docs.ultralytics.com/reference/utils/tuner/#ultralytics.utils.tuner.run_ray_tune
def get_args_parse():
    parser = argparse.ArgumentParser("Exploratory Data Analysis")
    parser.add_argument("--data", default="/hpc/home/csr33/ast_object_detection/ast.yaml", type=str)
    parser.add_argument("--model", default='yolov8n.pt', type=str)
    args = parser.parse_args()
    return args

# Initialize Your wandb Project: wandb will automatically detect the API key and proceed with the setup.
wandb.init()

# Define the Sweep: 
#A Sweep combines a strategy for trying out a bunch of hyperparameter values with the code that evaluates them. You need to define your strategy in the form of a configuration.
sweep_config = {
    'method': 'bayes',  # grid, random
     "name": "sweep",
    'metric': {'name': 'loss', 'goal': 'minimize'}, #validation loss, accuracy, val_acc
    'parameters': {'learning_rate': {'min': 0.0001, 'max': 0.1},
                   'batch_size': {'values': [2, 4, 8, 16, 32, 64, 128]},
                   "epochs": {"min": 1, 'max': 100},
                   'optimizer': {'values': ['adam', 'sgd']},
                   'fc_layer_size': {'values': [128, 256, 512]},
                   'dropout': {'values': [0.3, 0.4, 0.5]},
                   }
}


#Initialize the Sweep: Use wandb to initialize the sweep with your configuration.
sweep_id = wandb.sweep(sweep=sweep_config, project="Optimizing Hyperparameters for Yolov8")


#Define Your Training Procedure: This is where you define your model, loss function, optimizer, and other parameters. You'll also log your metrics with wandb.log().

def train():
    # default hyperparameters
    config_defaults = {'learning_rate': 0.01, 'batch_size': 64, 
                       "epochs":50, "optimizer":"adam", 
                      'fc_layer_size':256,'dropout':0.4}

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config

    # Define the model, loss and optimizer
    model = YOLO(args.model)
    criterion = ...
    optimizer = ...

    for epoch in range(epochs):
        output = model(input)
        loss = criterion(output, target)
        
        # Log the loss to wandb
        wandb.log({"loss": loss})

        loss.backward()
        optimizer.step()

- Start the Sweep: Finally, use wandb.agent() to start the sweep.

wandb.agent(sweep_id, train)

This will start the hyperparameter optimization process. wandb's Sweeps will automatically search through combinations of hyperparameter values (e.g. learning rate, batch size) to find the most optimal values.
Remember, hyperparameter tuning is an iterative process aimed at optimizing the machine learning model's performance metrics, such as accuracy, precision, and recall.
I hope this helps! Let me know if you have any other questions. 😊



# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize AST Hyperparameters",
    "spec": {"maxCombo": 5, "objective": "minimize", "metric": "loss"},
    "parameters": {
        "first_layer_units": {
            "type": "integer",
            "mu": 500,
            "sigma": 50,
            "scalingType": "normal",
        },
        "batch_size": {"type": "discrete", "values": [2, 4, 8, 16, 32, 64, 128, 256]},
    },
    "trials": 1,
}

opt = comet_ml.Optimizer(config)
def tune(args):
    
    comet_ml.init(project_name="yolov8-ast")

    # Load the model.
    # Initialize the YOLO model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(data=args.data, epochs=30, iterations=300, optimizer='AdamW',
               plots=False, save=False, val=False,
               name=f'{os.path.splitext(args.model)[0]}_tune')


if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    tune(args)
