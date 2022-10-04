import requests
import json
import os
from glob import glob
import tqdm
import copy
import numpy as np
import pandas as pd
import geopandas as gpd #important
from shapely.geometry import Polygon, Point
import volume_estimation_functions as vol_est
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser(description='This script adds LPC data to tile level tank data')
    parser.add_argument('--tnm_url', type=str, default='https://tnmaccess.nationalmap.gov/api/v1/products',
                        help='tnm api url')
    parser.add_argument('--tile_level_annotations_path', type=str, default="//oit-nas-fe13dc.oit.duke.edu//data_commons-borsuk//complete_dataset_/tile_level_annotations/tile_level_annotations.geojson", 
                        help='path to hold, lidar by tank geojson data')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='name of dataset type to pull from api, sgDataset format')
    parser.add_argument('--stored_data_path', type=str, default=None,
                        help='path to save requested tnm data, sgDataset format')
    parser.add_argument('--request_total_idx', type=str, default=None, 
                        help="idx for request enteries")
    parser.add_argument('--request_content_idx', type=str, default=None,
                        help='idx that holds contents in request')
    parser.add_argument('--request_content_names_idx', type=str, default=None, 
                        help="idx that holds the dataset name")
    args = parser.parse_args()
    return args

def main(args):
    tile_level_annotations = gpd.read_file(args.tile_level_annotations_path)
    complete_df = vol_est.usgs_api(tile_level_annotations, args.tnm_url, args.dataset_name, args.request_total_idx, 
                                       args.request_content_idx, args.request_content_names_idx)
    
    complete_df.to_file(os.path.join(args.stored_data_path, args.dataset_name + "subset.geojson"), driver="GeoJSON")


if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
