#help('modules')
import os
import json
from glob import glob
import re
import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd #important
import rasterio

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar # For dealing with Colorbars the proper way - TBD in a separate PyCoffee ?

import volume_estimation_functions as vol_est
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser(description='This script adds LPC data to tile level tank data')
    
    parser.add_argument('--tank_ids', type=str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk/complete_dataset/tiles", 
                        help='tank ids list')
    
    parser.add_argument('--lidar_path_by_tank_for_height', type=str, default = None, 
                        help='file path to list of files of type geojson to lidar data for each tank')
    parser.add_argument('--lidar_by_tank_dir', type=str, default = None, 
                        help='directory to lidar by tanks')
    
    parser.add_argument('--DEM_path_by_tank_for_height', type=str, default = None, 
                        help='file path to list of files of type tif to DEM data for each tank')
    parser.add_argument('--DEM_by_tank_dir', type=str, default = None, 
                        help='directory to DEM by tanks')   
    
    parser.add_argument('--aerial_image_path_by_tank_for_height', type=str, default = None, 
                        help='file path to list of files of type jpg to aerial image data for each tank')
    parser.add_argument('--image_by_tank_dir', type=str, default = None, 
                        help='directory to image by tanks')  

    parser.add_argument('--tiles_dir', type=str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk/complete_dataset/tiles", 
                        help='tile level tank annotations')
    
    parser.add_argument('--plot_dir', type=str, default = None, 
                        help='folder to hold plots')
    args = parser.parse_args()
    return args

def main(args):
    #read in tile level annotations
    #rite variables needed 
    tank_ids = vol_est.read_list(args.tank_ids)
    
    #read in list of lidar datasets
    if type(args.lidar_path_by_tank_for_height) == type(None):
        lidar_path_by_tank_for_height = glob(args.lidar_by_tank_dir)
    else:
        lidar_path_by_tank_for_height = vol_est.read_list(args.lidar_path_by_tank_for_height)
    
    #read in list of DEM datasets
    if type(args.DEM_path_by_tank_for_height) == type(None):
        DEM_path_by_tank_for_height = glob(args.DEM_by_tank_dir)
    else:
        DEM_path_by_tank_for_height = vol_est.read_list(args.DEM_path_by_tank_for_height)
    
    #read in list of image datasets
    if type(args.aerial_image_path_by_tank_for_height) == type(None):
        aerial_image_path_by_tank_for_height = glob(args.image_by_tank_dir)
    else:
        aerial_image_path_by_tank_for_height = vol_est.read_list(args.aerial_image_path_by_tank_for_height)
        
    vol_est.height_estimation_figs(tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height, aerial_image_path_by_tank_for_height,
                                   args.plot_dir, args.tiles_dir)
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


