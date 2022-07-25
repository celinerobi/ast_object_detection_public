print("load modules")
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

print("dem by tank")
def get_args_parse():
    parser = argparse.ArgumentParser(description='This script adds LPC data to tile level tank data')
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
    os.makedirs(args.dem_EPSG4326_path, exist_ok = True)
    os.makedirs(args.DEM_by_tank_output_path, exist_ok = True)
    
    print("reproject dem")
    vol_est.reproject_dems(args.dem_path, args.dem_EPSG4326_path) #reproject DEM
    projected_dem_paths = glob(args.dem_EPSG4326_path + "/*.tif") #get path of dems
    print(projected_dem_paths)
    #specify output path
    vol_est.dem_by_tank(projected_dem_paths, tank_data, args.DEM_by_tank_output_path)
    lidar_path_by_tank_for_height, DEM_path_by_tank_for_height = vol_est.identify_tank_ids(args.lidar_by_tank_output_path, args.DEM_by_tank_output_path)
    print(lidar_path_by_tank_for_height)
    print(DEM_path_by_tank_for_height)
    vol_est.add_bare_earth_data_to_lpc_by_tank_data(lidar_path_by_tank_for_height, DEM_path_by_tank_for_height)
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)


