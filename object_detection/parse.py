import json
import os

import random 
import math

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT

from PIL import Image
import xml.etree.ElementTree as et

import argparse


import dataset

#python parse.py --parent_directory ~/work/Test --img_directory chips_positive --annotation_directory chips_positive_xml --path_to_predefined_classes ~/work/AST/object_detection/predefined_classes.txt 

#python parse.py --parent_directory /shared_space/natech/Test --img_directory chips_positive --annotation_directory chips_positive_xml --path_to_predefined_classes ~/work/AST/object_detection/predefined_classes.txt --train_val_percent .4

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--parent_directory', type=str, default=None,
                        help='path to parent directory, holding the images + annotation directory.')
    parser.add_argument('--img_directory', type=str, default=None,
                        help='path to image directory, holds images.')
    parser.add_argument('--annotation_directory', type=str, default=None,
                        help='path to annotation directory, holding the xml files.')
    parser.add_argument('--complete_img_ids', type=str, default=None,
                        help='The text file associated with all of the image ids.')
    parser.add_argument('--path_to_predefined_classes', type=str, default=None,
                        help='The text file containing a list of the predefined classes')    
    parser.add_argument('--train_val_percent', type=float, default=0.8,
                        help='The percent of the data seperated into the train/val set')  
    parser.add_argument('--bbox_remove', type=int, default=20,
                        help='The pixel wideth/height to remove bboxes')  
    
    parser.add_argument('--train_val_percent', type=float, default=0.8,
                        help='The percent of the data seperated into the train/val set')  
    parser.add_argument('--bbox_remove', type=int, default=20,
                        help='The pixel wideth/height to remove bboxes')  
    args = parser.parse_args()
    return args

def main(args):  
    if args.complete_img_ids == None:
        dataset.make_list_of_image_ids(args.parent_directory, args.img_directory, args.annotation_directory)
        dataset.split_train_val_test(args.parent_directory, "img_ids.txt", args.train_val_percent)
    else:
        dataset.split_train_val_test(args.parent_directory, args.complete_img_ids, args.train_val_percent)
    n_test_images, n_test_objects, path = dataset.create_data_lists(args.parent_directory, args.img_directory, 
                                                                    args.annotation_directory, args.path_to_predefined_classes, 
                                                                    "test_img_id.txt", "test", args.bbox_remove)
        
    n_train_val_images, n_train_val_objects, path = dataset.create_data_lists(args.parent_directory, args.img_directory, 
                                                                              args.annotation_directory, args.path_to_predefined_classes,
                                                                              "train_val_img_id.txt", "train", args.bbox_remove)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
           n_test_images, n_test_objects, path))
    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
           n_train_val_images, n_train_val_objects, path))
    
if __name__ == '__main__':
    # Set fixed random number seed

    args = get_args_parser()
    main(args)
