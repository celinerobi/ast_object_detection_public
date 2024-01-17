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

def train(args):
    comet_ml.init(project_name="yolov8-ast")

    # Load the model.
    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

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
