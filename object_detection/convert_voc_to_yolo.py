import os
import glob
import os
import argparse
import pickle
import xml.etree.ElementTree as ET
from os.path import join
import math
import random
import shutil
#modules in files
import dataset

# based on https://gist.github.com/M-Younus/ceaf66e11a9c0f555b66a75d5b557465
def paths_to_ids(img_paths):
    img_ids = [] 
    for img_path in img_paths:
        basename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        img_ids.append(basename_no_ext)
    return(img_ids)

def write_list(destination_path, list_):
    f = open(destination_path, 'w')
    f.writelines(s + ' \n' for s in list_)
    f.close() #to change file access modes
    
def read_list(txt_file):
    f = open(txt_file, "r")
    list_ = f.read()
    f.close()
    return(list_.split("\n"))
    
def make_split_dirs(data_folder):
    #make directories to store split data
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(os.path.join(data_folder, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(data_folder, "test", "labels"), exist_ok=True)

def get_image_paths_in_dir(dir_path, save_dir):
    """Function to get the image paths from a given directory 
    Args:
        dir_path (str): directory that contains labeled (annotated) images
        save_dir (str): path to dir to save outputs

    Returns:
        img_paths: List of images in specified directory
        img_ids: List image names with the extension removed (referred to as image ids)
    """
    img_paths = []
    for img_path in glob.glob(dir_path + '/*.jpg'):
        img_paths.append(img_path)
    write_list(os.path.join(save_dir,"img_path.txt"), img_paths)
    
    img_ids = paths_to_ids(img_paths)
    write_list(os.path.join(save_dir,"img_ids.txt"), img_ids)

    return(img_paths, img_ids)

def split_train_val_test(img_paths, seed, data_folder, save_dir,
                         train_percent = 0.8, val_percent = 0.1):
    """
    get a text file of the ids for the train/val and test sets
    Percentage of trainval:test and train: validation 
    """
    #Calculate numbers of images that should be in the train/val and test sets   
    num_imgs = len(img_paths)  
    num_train_imgs = math.ceil(num_imgs * train_percent)  
    num_val_imgs = math.ceil(num_train_imgs * val_percent)  
    num_test_imgs = num_imgs - num_train_imgs
    #randomly sample images to be in the train/val and test sets
    random.seed(1)
    train_img_paths = random.sample(img_paths, num_train_imgs)
    val_img_paths = random.sample(train_img_paths, num_val_imgs)  
    test_img_paths = list(set(img_paths) - set(train_img_paths) - set(val_img_paths))    
 
    train_img_ids = paths_to_ids(train_img_paths)
    val_img_ids = paths_to_ids(val_img_paths)
    test_img_ids = paths_to_ids(test_img_paths)    
    #Write .txt files.
    write_list(os.path.join(save_dir,"train_img_paths.txt"), train_img_paths)
    write_list(os.path.join(save_dir,"val_img_paths.txt"), val_img_paths)
    write_list(os.path.join(save_dir,"test_img_paths.txt"), test_img_paths)
    write_list(os.path.join(save_dir,"train_img_ids.txt"), train_img_ids)
    write_list(os.path.join(save_dir,"val_img_ids.txt"), val_img_ids)
    write_list(os.path.join(save_dir,"test_img_ids.txt"), test_img_ids)

    # Copy-pasting images
    for i, (img_path, img_id) in enumerate(zip(train_img_paths, train_img_ids)):
        shutil.copy(img_path, os.path.join(data_folder, "train", "images", img_id+".jpg"))
        xml_path = img_path.replace('chips_positive', 'chips_positive_xml')
        xml_path = xml_path.replace("jpg","xml")
        shutil.copy(xml_path, os.path.join(data_folder, "train", "labels", img_id+".xml"))
        
    for i, (img_path, img_id) in enumerate(zip(val_img_paths, val_img_ids)):
        shutil.copy(img_path, os.path.join(data_folder, "val", "images", img_id+".jpg"))
        xml_path = img_path.replace('chips_positive', 'chips_positive_xml')
        xml_path = xml_path.replace("jpg","xml")
        shutil.copy(xml_path, os.path.join(data_folder, "val", "labels", img_id+".xml"))
        
    for i, (img_path, img_id) in enumerate(zip(test_img_paths, test_img_ids)):
        shutil.copy(img_path, os.path.join(data_folder, "test", "images", img_id+".jpg"))
        xml_path = img_path.replace('chips_positive', 'chips_positive_xml')
        xml_path = xml_path.replace("jpg","xml")
        shutil.copy(xml_path, os.path.join(data_folder, "test", "labels", img_id+".xml"))
    
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(img_id, dir_path, voc_data_dir, split, coco_data_dir, label_map):
    """
    Excludes objects labeled difficult from conversion 
    """
    voc_path = os.path.join(dir_path, voc_data_dir, split, "labels", img_id+".xml")
    in_file = open(voc_path)
    #specify path for coco formated annotations
    coco_path = voc_path.replace(voc_data_dir, coco_data_dir)
    coco_path = coco_path.replace("xml", "txt")
    out_file = open(coco_path, 'w')
    
    #parse files and re
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in label_map or int(difficult)==1:
            continue
        cls_id = label_map[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
      
    
def copy_converted_images(img_id, dir_path, voc_data_dir, split, coco_data_dir):
    """
    Excludes objects labeled difficult from conversion 
    """
    voc_img_path = os.path.join(dir_path, voc_data_dir, split, "images", img_id+".jpg")
    coco_img_path = os.path.join(dir_path, coco_data_dir, split, "images", img_id+".jpg")
    shutil.copy(voc_img_path, coco_img_path)

        
def get_args_parse():
    parser = argparse.ArgumentParser(
        description='This script adds a subdirectory of xmls to correct possible inconsistent labels')
    parser.add_argument('--dir_path', type = str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk//complete_dataset",
                        help = 'path to parent directory; the directory of the storge space.')
    parser.add_argument('--save_dir', type = str, 
                        help = 'path to the verified complete dataset.')
    parser.add_argument('--voc_data_folder', type = str,
                        help = 'The file path of the numpy array that contains the image tracking.')
    parser.add_argument('--coco_data_folder', type = str,
                        help = 'The file path of the numpy array that contains the image tracking.')
    parser.add_argument('--path_to_predefined_classes', type = str, 
                        help = 'The file path of the numpy array that contains the tile names and tile urls of the complete arrays.')
    args = parser.parse_args()
    return args

def main(args):
    #make directories to store split data
    make_split_dirs(args.voc_data_folder)    
    make_split_dirs(args.coco_data_folder)
    
    img_paths, img_ids = get_image_paths_in_dir(args.dir_path, args.save_dir)
    #split_train_val_test(img_paths, 1, args.voc_data_folder, args.save_dir, train_percent = 0.8, val_percent = 0.1)
    label_map = dataset.get_label_map(args.save_dir, args.path_to_predefined_classes)
    for img_id in read_list(os.path.join(args.save_dir,"train_img_ids.txt")):
        convert_annotation(img_id, args.dir_path, args.voc_data_folder, "train", args.coco_data_folder, label_map)
        copy_converted_images(img_id, args.dir_path, args.voc_data_folder, "train", args.coco_data_folder)
                            
    for img_id in read_list(os.path.join(args.save_dir,"val_img_ids.txt")):
        convert_annotation(img_id, args.dir_path, args.voc_data_folder,"val", args.coco_data_folder, label_map)
        copy_converted_images(img_id, args.dir_path, args.voc_data_folder, "val", args.coco_data_folder)
    for img_id in read_list(os.path.join(args.save_dir,"test_img_ids.txt")):
        convert_annotation(img_id, args.dir_path, args.voc_data_folder, "test", args.coco_data_folder, label_map)
        copy_converted_images(img_id, args.dir_path, args.voc_data_folder, "test", args.coco_data_folder)


if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
