import tqdm
import numpy as np
import pandas as pd
import os
from glob import glob

import geopandas as gpd #important
import volume_estimation_functions as vol_est
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser(description='Get the aerial data for each tank')
    parser.add_argument('--tank_data_path', type=str, default="//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk/complete_dataset/tile_level_annotations/tile_level_annotations.geojson",
                        help='tank dataset path')
    parser.add_argument('--tiles_dir', type=str, default = "//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk/complete_dataset/tiles", 
                        help='tile level tank annotations')
    parser.add_argument('--tank_imagery_dir', type=str, default="//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk/aerial_images_by_tank",
                        help='path to hold by tank imagery data')
    args = parser.parse_args()
    return args

def main(args):
    #read tank dataset
    tank_data = gpd.read_file(args.tank_data_path)
    tank_data = tank_data.to_crs('EPSG:4326')
    tank_data.crs = "EPSG:4326" #assign projection
    #get images path list
    tank_imagery_paths = vol_est.image_by_tank(tank_data, args.tiles_dir, args.tank_imagery_dir)

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
