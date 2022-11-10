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
    parser.add_argument('--tank_data_path', type=str,
                        default="/hpc/group/borsuklab/csr33/volume_estimation/tile_level_annotations/tile_level_annotations.geojson",
                        help='path to hold, lidar by tank geojson data')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='name of dataset type to pull from api, sgDataset format')
    parser.add_argument('--dataset_abbrv', type=str, default=None,
                        help='name of dataset type to pull from api, abbrv format')
    parser.add_argument('--stored_data_path', type=str,
                        default='/hpc/group/borsuklab/csr33/volume_estimation/usgs_tnm_api',
                        help='path to save requested tnm data, sgDataset format')
    parser.add_argument('--request_total_idx', type=str, default='total',
                        help="idx for the number of requested enteries")
    parser.add_argument('--request_content_idx', type=str, default='items',
                        help='idx that holds contents in request')
    parser.add_argument('--request_content_names_idx', type=str, default='title',
                        help="idx that holds the dataset name")
    args = parser.parse_args()
    return args

def main(args):
    tank_data = gpd.read_file(args.tank_data_path)
    complete_df = vol_est.usgs_api(tank_data, args.tnm_url, args.dataset_name, args.request_total_idx,
                                       args.request_content_idx, args.request_content_names_idx)

    complete_df.to_file(os.path.join(args.stored_data_path, args.dataset_abbrv + "_subset.geojson"), driver="GeoJSON")

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
