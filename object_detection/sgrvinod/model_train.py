import json
import os

import random 
import math
import numpy as np

import time
from datetime import datetime, timedelta

from PIL import Image
import xml.etree.ElementTree as et

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as FT
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import argparse
import sgrvinod

from sgrvinod import train
from sgrvinod import evaluate

import utils
import transforms

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='This script is used to train the model')
    parser.add_argument('--parent_directory', type=str, default=None,
                        help='path to parent directory, holding the annotation directory.')
    parser.add_argument('--path_to_predefined_classes', type=str, default=None,
                        help='The text file containing a list of the predefined classes')   
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Whether or not to use the the pretrained model')
        
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='The percentage of the data allocated to the validation set')
    
    parser.add_argument('--keep_difficult', type=bool, default=True,
                        help='Whether or not to difficult objects')
    
    parser.add_argument('--scheduler_name', type=str, default='cosineannealinglr',
                        help='learning rate scheduler name')
    parser.add_argument('--optimizer_name', type=str, default='SGD',
                        help='optimizer name')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate.')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--milestones', type=int, default=5,
                        help='milestones.')
    parser.add_argument('--T_max', type=float, default=0.1,
                        help='T_max')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight_decay')
    
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='num_epochs')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num_workers')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='The batch size.')
    args = parser.parse_args()
    return args 

def main(args):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Data loading code
    print("Loading data")
    train_images, train_objects, val_images, val_objects, test_images, test_objects = utils.split_method(args.parent_directory, "simple_val",  val_size = args.val_size)
    
    train_dataset = utils.pascal_voc_dataset(train_images, train_objects, train = True)
    val_dataset = utils.pascal_voc_dataset(val_images, val_objects, train = False)
    test_dataset = utils.pascal_voc_dataset(test_images, test_objects, train = False)

    # Custom dataloaders
    print("Creating data loaders")
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, 
                                                    num_workers=args.num_workers, collate_fn = train_dataset.collate_fn, pin_memory=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,  batch_size = args.batch_size, shuffle=True, 
                                                  num_workers=args.num_workers, collate_fn = val_dataset.collate_fn, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,  batch_size = args.batch_size, shuffle=True, 
                                                  num_workers=args.num_workers, collate_fn = val_dataset.collate_fn, pin_memory=True)
    
    print("Creating model")
    # Model parameters
    label_map = utils.get_label_map(args.parent_directory, args.path_to_predefined_classes)
    label__color_map = utils.get_label_color_map(label_map)
    # replace the classifier with a new one, that has the num_classes which is user-defined
    num_classes= len(label_map)  # number of different types of objects

    # get number of input features for the classifier
    model = sgrvinod.train.get_frcnn_model(num_classes, args.pretrained)
    model.to(device)
    
    ## Define make_optimizer() and make_scheduler()
    optimizer = sgrvinod.train.make_optimizer(args.optimizer_name, model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = sgrvinod.train.make_scheduler(args.scheduler_name, optimizer, milestones=[args.milestones], lr_gamma = args.lr_gamma)


    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations 
        sgrvinod.train.train(train_loader=train_data_loader,
                       model=model,
                       optimizer=optimizer,
                       epoch=epoch,
                       device=device,
                       print_freq = 10)
        
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        sgrvinod.evaluate.evaluate(val_data_loader, model.eval(), device = device)

        # Save checkpoint
        #save_checkpoint(epoch, model, optimizer)

if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(1)
    args = get_args_parser()
    main(args)