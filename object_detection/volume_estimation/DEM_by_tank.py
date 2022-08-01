#help('modules')

import os
from glob import glob
import re
import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd #important
import rasterio
import volume_estimation_functions as vol_est
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser(description='This script adds LPC data to tile level tank data')
    parser.add_argument('--tile_level_annotation_path', type=str, default=None, 
                        help='tile level tank annotations')
    parser.add_argument('--lidar_by_tank_output_path', type=str, default=None, 
                        help='path to folder where lidar by tank can be stored')
    parser.add_argument('--dem_path', type=str, default=None,
                        help='path to folder containing de, in study area (tif file)')
    parser.add_argument('--dem_EPSG4326_path', type=str, default=None, 
                        help='path to hold projected DEM (EPSG4326)')
    parser.add_argument('--DEM_by_tank_output_path', type=str, default=None, 
                        help='path to folder where DEM by tank can be stored')
    args = parser.parse_args()
    return args

def main(args):
    #read in tile level annotations
    tile_level_annotation_path = gpd.read_file(args.tile_level_annotation_path)

    #read and reproject DEM
    os.makedirs(args.dem_EPSG4326_path, exist_ok = True)
    os.makedirs(args.DEM_by_tank_output_path, exist_ok = True)
    vol_est.reproject_dems(args.dem_path, args.dem_EPSG4326_path) #reproject DEM
    projected_dem_paths = glob(args.dem_EPSG4326_path + "/*.tif") #get path of dems
    
    #clip dem to tank
    vol_est.dem_by_tank(projected_dem_paths, tile_level_annotation_path, args.DEM_by_tank_output_path)
    #identify tanks that have lidar and tank data 
    tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height = vol_est.identify_tank_ids(args.lidar_by_tank_output_path, args.DEM_by_tank_output_path)
    vol_est.add_bare_earth_data_to_lpc_by_tank_data(tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height)
    
    #rite variables needed 
    vol_est.write_list(tank_ids, "tank_ids.json")
    vol_est.write_list(lidar_path_by_tank_for_height, "lidar_path_by_tank_for_height.json")
    vol_est.write_list(DEM_path_by_tank_for_height, "DEM_path_by_tank_for_height.json")
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


