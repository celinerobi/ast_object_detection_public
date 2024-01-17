import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import json
random.seed(108)


def get_label_map(save_dir, path_to_predefined_classes):
    """
    get the label map for the dataset
    """
    with open(os.path.join(path_to_predefined_classes)) as f:
        labels = f.read().splitlines()
        label_map = {k: v for v, k in enumerate(labels)}
    with open(os.path.join(save_dir, "label_map.json"), 'w') as j:
        json.dump(label_map, j)  # save label map too
    return label_map


def get_label_color_map(label_map):
    """
    get the label color map for the dataset
    """
# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8',
                       '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080',
                       '#aa6e28', '#fffac8', '#800000', '#aaffc3']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
    return label_color_map


# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolo(info_dict, annotations_path_yolo):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join(annotations_path_yolo, info_dict["filename"]+".txt")
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
    
    
def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()
    
    
def test_annotations(annotations, class_name_to_id_mapping):
    #Gut Check: check to make sure new annotatiosn are correct 
    #Get any random annotation file 
    random.seed(0)
    class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))
    annotation_file = random.choice(annotations)
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace("labels", "images").replace("txt", "jpg")
    #image_file = '/work/csr33/ast/complete-dataset/chips_positive/'+os.path.basename(annotation_file).replace("txt", "jpg")
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_bounding_box(image, annotation_list)


#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False
            
            
def main(args):
    save_dir="/work/csr33/ast"
    path_to_predefined_classes="/work/csr33/ast/predefined_classes.txt"
    # Get the annotations
    original_data_dir = "/work/csr33/ast/complete-dataset"
    original_data_labels_folder = "chips_positive_corrected_xml"
    original_data_imgs_folder = "chips_positive"

    yolo_dir = "/work/csr33/ast/datasets/yolo"

    # Set the split ratios (adjust as needed)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    
    #copy images from complete_dataset to yolov5 datadir
    yolo_dir_temp = os.path.join(yolo_dir, "temp")
    os.makedirs(yolo_dir_temp, exists_ok=True)
    shutil.copytree(os.path.join(original_data_dir, original_data_imgs_folder),
                    os.path.join(yolo_dir_temp, "images"))
    #!cp -r "/work/csr33/ast/complete-dataset/chips_positive/." "/work/csr33/ast/complete-dataset-yolov5/images"

    # Dictionary that maps class names to IDs
    class_name_to_id_mapping = get_label_map(save_dir, path_to_predefined_classes)

    #get list of annotations in original dir
    annotations = glob(os.path.join(original_data_dir, original_data_labels_folder) + "/*.xml")
    annotations.sort()

    # Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolo(info_dict, os.path.join(yolo_dir_temp, "labels"))
    
    #replace list of annotations with yolov
    annotations = glob(os.path.join(yolo_dir_temp, "labels") + "/*.txt")
    images = glob(os.path.join(yolo_dir_temp, "images") + "/*.jpg")
    images.sort()
    annotations.sort()
    
    #Next we partition the dataset into train, validation, and test sets containing 80%, 10%, and 10% of the data, respectively. 
    #You can change the split values according to your convenience.
    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = (1 - train_ratio), random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = (test_ratio / (1 - train_ratio)), random_state = 1)

    #Make split directories
    data_type = ["images", "labels"]
    split = ["train", "val", "test"]
    for d in data_type:
        for s in split:
            directory_path =  os.path.join(yolo_dir, d, s)
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

                
    #Move the files to their respective folders.
    # Move the splits into their folders
    move_files_to_folder(train_images, os.path.join(yolo_dir, "images", "train"))
    move_files_to_folder(val_images, os.path.join(yolo_dir, "images", "val"))
    move_files_to_folder(test_images, os.path.join(yolo_dir, "images", "test"))
    move_files_to_folder(train_annotations, os.path.join(yolo_dir, "labels", "train"))
    move_files_to_folder(val_annotations, os.path.join(yolo_dir, "labels", "val"))
    move_files_to_folder(test_annotations, os.path.join(yolo_dir, "labels", "test"))