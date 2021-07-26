import json
import os

import random 
import math

import torch
import torchvision
import torchvision.transforms.functional as FT
from sklearn.model_selection import KFold, train_test_split

from PIL import Image
import xml.etree.ElementTree as et

from transforms import get_transform

"""
Specify the label/ label color map
"""       
def get_label_map(parent_dir, path_to_predefined_classes):
    """
    get the label map for the dataset
    """
    with open(os.path.join(path_to_predefined_classes)) as f:
        labels = f.read().splitlines()
        label_map = {k: v + 1 for v, k in enumerate(labels)}
        label_map['background'] = 0
    with open(os.path.join(parent_dir, "label_map.json"), 'w') as j:
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

"""
get the image id's for all of the dataset and the train/val + test set
"""
def make_list_of_image_ids(parent_dir, img_dir = "img", anno_dir = "xml"):
    """
    parameters: the parent_directory, and the sub directories containing the images + annotations
    returns: text file of the ids for all of images in the datasets
    """   
    img_path = os.path.join(parent_dir, img_dir)
    anno_path = os.path.join(parent_dir, anno_dir)
    
    #if not (os.path.isdir(img_path) | os.path.isdir(anno_path)):
    #    print( "directories not found")
    assert (os.path.isdir(img_path) | os.path.isdir(anno_path)), "directories not found"

    img_files = os.listdir(img_path) #pull the files in the img folder
    img_id = []
    for img_file in img_files:
        img_id.append(''.join(map(str,[img_file.split(".",1)[0],"\n"])))
        
    f = open(os.path.join(parent_dir, "img_ids"+'.txt'), 'w')
    f.writelines(img_id)
    f.close() #to change file access modes

    
def split_train_val_test(parent_directory, complete_img_ids, train_val_percent = 0.8):
    """
    get a text file of the ids for the train/val and test sets
    Percentage of trainval:test and train: validation 
    """
    #get list of all of the img_ids
    with open(os.path.join(parent_directory, complete_img_ids)) as f:
        img_ids_list = f.read().splitlines()

    #Calculate numbers of images that should be in the train/val and test sets
    num_imgs = len(img_ids_list)  
    num_train_val_imgs = math.ceil(num_imgs * train_val_percent)  
    num_test_imgs = num_imgs - num_train_val_imgs

    #randomly sample images to be in the train/val and test sets
    random.seed(42)
    train_val_img_idx = random.sample(range(num_imgs), num_train_val_imgs)  
    test_img_idx = list(set(list(range(num_imgs))) - set(train_val_img_idx))

    #get a list of the values for the randomly sampled values
    train_val_img_id_list = []
    test_img_id_list = [] 

    for index, value in enumerate(img_ids_list):
        if index in train_val_img_idx:  
            train_val_img_id_list.append(''.join(map(str,[value,"\n"])))

        if index in test_img_idx:  
            test_img_id_list.append(''.join(map(str,[value,"\n"])))

    #Write .txt files.
    with open(os.path.join(parent_directory,"train_val_img_id.txt"), 'w') as train_val_img_id:
        train_val_img_id.writelines(train_val_img_id_list) #write lines
    with open(os.path.join(parent_directory,"test_img_id.txt"), 'w')  as test_img_id:
        test_img_id.writelines(test_img_id_list)

"""
This parses the data downloaded and saves the following files –

A JSON file for each split with a list of the absolute filepaths of I images, where I is the total number of images in the split.

A JSON file for each split with a list of I dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties. The ith dictionary in this list will contain the objects present in the ith image in the previous JSON file.

A JSON file which contains the label_map, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in utils.py and directly importable.
"""
def parse_annotation(anno_path, img_id, label_map, keep_difficult = True, bbox_remove = 20):
    """
    for each img, parse the annotations in a format readable for pytorch
    argument: the parent directory and subdirectory containing the image; the imageid for the image of interest; the label map
    returns: a dictionary containing the bounding boxes, labels, and difficults
    """
    tree = et.parse(os.path.join(anno_path, img_id +".xml"))
    root = tree.getroot()

    boxes = []
    labels = []
    difficulties = []
    
    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1')
        label = object.find('name').text.lower().strip()
        if label not in label_map:
            print(label)
            continue

        bbox = object.find('bndbox')
        xmin = float(bbox.find('xmin').text) - 1
        ymin = float(bbox.find('ymin').text) - 1
        xmax = float(bbox.find('xmax').text) - 1
        ymax = float(bbox.find('ymax').text) - 1
        
        remove_bbox = (xmax == xmin) | (ymax == ymin) | (((xmax - xmin) <= bbox_remove) & ((ymax - ymin) <= bbox_remove))
        if remove_bbox:
            continue
        elif not keep_difficult:
            continue
        else:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[label])
            difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(parent_dir, img_dir, anno_dir, path_to_predefined_classes, subset_img_ids, subset_name, bbox_remove = 20):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    parent_directory, subset_img_ids, subset_name
    """

    img_path = os.path.join(parent_dir, img_dir)
    anno_path = os.path.join(parent_dir, anno_dir)

    """#specify subdirectory
    for folder in os.listdir(parent_directory): #identifies the subfolders
        d = parent_directory + "/"+ folder #creates the complete path for each subfolder
        if os.path.isdir(d):
            sub_directory = d #adds the subfolder to the list
    """
    #initial list and counter
    subset_images = []
    subset_objects = []
    empty_images = []
    n_objects = 0
    no_obs = 0
    # Find IDs of images for subset
    with open(os.path.join(parent_dir, subset_img_ids)) as f:
        ids = f.read().splitlines()
        
    for id in ids:
        # Parse annotation's XML file
        objects = parse_annotation(anno_path, id, get_label_map(parent_dir, path_to_predefined_classes),
                                    keep_difficult = True, bbox_remove = bbox_remove)
        if len(objects['boxes']) == 0:
            no_obs += 1
            empty_images.append(os.path.join(img_path, id + '.jpg'))
            continue
        n_objects += len(objects['boxes'])
        subset_objects.append(objects)
        subset_images.append(os.path.join(img_path, id + '.jpg'))
    assert len(subset_objects) == len(subset_images)
    
    # Save to file
    with open(os.path.join(parent_dir, subset_name + '_images.json'), 'w') as j:
        for chunk in json.JSONEncoder().iterencode(subset_images):
            j.write(chunk)
    with open(os.path.join(parent_dir, subset_name + '_objects.json'), 'w') as j:
        for chunk in json.JSONEncoder().iterencode(subset_objects):
            j.write(chunk)
    with open(os.path.join(parent_dir, 'empty_images.json'), 'w') as j:
        for chunk in json.JSONEncoder().iterencode(empty_images):
            j.write(chunk)
            
    return len(subset_images), n_objects, os.path.abspath(parent_dir)
        
        
"""
DataLoader
"""

def split_method(parent_directory, method, val_size = 0.1):
    "Load the training data, and split out validation data"
    if method == "simple_val":
        
        for split in ["train", "test"]:
            with open(os.path.join(parent_directory, split + '_images.json'), 'r') as j:
                images = json.load(j)
            with open(os.path.join(parent_directory, split + '_objects.json'), 'r') as j:
                objects = json.load(j)
                assert len(images) == len(objects) 
            if split == "train":
                train_images, val_images, train_objects, val_objects = train_test_split(images, objects, test_size = val_size, random_state=42)
            else:
                test_images = images
                test_objects = objects

        return  train_images, train_objects, val_images, val_objects, test_images, test_objects
   # if Kfold:
        

class pascal_voc_dataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, images, objects, train):
        """
        The __init__ function is run once when instantiating the Dataset object. 
        We initialize the directory containing the images, the annotations file, 
        and both transforms (covered in more detail in the next section).

        :param data_folder: folder where data files are stored
        :param split: split, one of 'train' or 'test'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """        
        self.images = images #the image lists
        
        self.objects = objects #the object lists
        
        if train: # Read in train data 
            self.split = "train"
        else: # Read in test data
            self.split = "test"

    def __getitem__(self, idx):
        """
        The __getitem__ function loads and returns a sample from the dataset at the given index idx. 
        Based on the index, it identifies the image’s location on disk, converts that to a tensor using read_image,
        retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable), 
        and returns the tensor image and corresponding label in a tuple.
        
        image: a PIL Image of size (H, W)
        target: a dict containing the following fields
        boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
        image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        """
        # Load image
        #print("print image name from loader", self.images[idx])
        image = Image.open(self.images[idx], mode='r')
        image = image.convert('RGB')
        image_id = torch.tensor([idx])
        
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[idx]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)
                    
        #Apply transformations
        image, boxes, labels, difficulties = get_transform(image, boxes, labels, difficulties, split = self.split)
        num_objs = boxes.size()[0]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["difficulties"] = difficulties
        target["area"] = area
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        return image, target
        #return image, boxes, labels, difficulties


    def __len__(self):
        """
        The __len__ function returns the number of samples in our dataset.

        """
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        targets = list()

        for b in batch:
            images.append(b[0])
            targets.append(b[1])

        images = torch.stack(images, dim=0)

        return images, targets  # tensor (N, 3, 300, 300), 3 lists of N tensors each 
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
        """



# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py