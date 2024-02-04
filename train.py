from ultralytics import YOLO
import comet_ml
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser("Exploratory Data Analysis")
    parser.add_argument("--data", default="/hpc/home/csr33/ast_object_detection/ast.yaml", type=str)
    parser.add_argument("--model", default='yolov8x.pt', type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()
    return args

def train(args):
    comet_ml.init(project_name="yolov8-ast")

    if args.name == None:
        args.name = f'{os.path.splitext(args.model)[0]}_e{args.epochs}_imgsz{args.imgsz}'
    # Load the model.
    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)
    # Training.
    results = model.train(project = args.project_name, name = args.name,
                          imgsz=args.imgsz)
if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    train(args)
