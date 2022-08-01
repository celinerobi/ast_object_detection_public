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
    parser.add_argument('--tiles_dir', type=str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk/complete_dataset/tiles", 
                        help='tile level tank annotations')
    parser.add_argument('--plot_dir', type=str, default = None, 
                        help='folder to hold plots')
    args = parser.parse_args()
    return args

def main(args):
    #read in tile level annotations
    #rite variables needed 
    tank_ids = vol_est.read_list("tank_ids.json")
    lidar_path_by_tank_for_height = vol_est.read_list("lidar_path_by_tank_for_height.json")
    DEM_path_by_tank_for_height = vol_est.read_list("DEM_path_by_tank_for_height.json")
    vol_est.height_estimation_figs(tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height, args.plot_dir, args.tiles_dir):
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


