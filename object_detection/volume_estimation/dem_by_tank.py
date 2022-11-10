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
    parser = argparse.ArgumentParser(description='This script adds DEM  data to lidar by tank data')
    parser.add_argument('--tank_data_path', type=str, default=None,
                        help='tile level tank annotations')
    parser.add_argument('--lidar_by_tank_dir', type=str, default=None,
                        help='directory to store lidar by tank')
    parser.add_argument('--dem_dir', type=str, default=None,
                        help='directory to store dem in study area (tif file)')
    parser.add_argument('--dem_EPSG4326_dir', type=str, default=None,
                        help='directory to store projected DEM (EPSG4326)')
    parser.add_argument('--dem_by_tank_dir', type=str, default=None,
                        help='directory to store DEM by tank')
    args = parser.parse_args()
    return args

def main(args):
    #read in tile level annotations
    tank_data = gpd.read_file(args.tank_data_path)

    #read and reproject DEM
    os.makedirs(args.dem_EPSG4326_dir, exist_ok=True)
    os.makedirs(args.dem_by_tank_dir, exist_ok=True)
    vol_est.reproject_dems(args.dem_dir, args.dem_EPSG4326_dir) #reproject DEM
    projected_dem_paths = glob(args.dem_EPSG4326_dir + "/*.tif") #get path of dems
    
    #clip dem to tank
    vol_est.dem_by_tank(projected_dem_paths, tank_data, args.dem_by_tank_dir)
    # identify tanks that have lidar and tank data
    tank_ids, lidar_path_by_tank_for_height, dem_path_by_tank_for_height \
        = vol_est.identify_tank_ids(args.lidar_by_tank_dir, args.dem_by_tank_dir)
    vol_est.add_bare_earth_data_to_lpc_by_tank_data(lidar_path_by_tank_for_height, dem_path_by_tank_for_height)
    
    #write variables needed
    vol_est.write_list(tank_ids, "tank_ids.json")
    vol_est.write_list(lidar_path_by_tank_for_height, "lidar_path_by_tank_for_height.json")
    vol_est.write_list(dem_path_by_tank_for_height, "DEM_path_by_tank_for_height.json")
    
if __name__ == '__main__':
    args = get_args_parse()
    main(args)


