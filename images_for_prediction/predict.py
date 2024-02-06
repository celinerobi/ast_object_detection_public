import os
import argparse
import math
import ast
from copy import copy

import pandas as pd
import numpy as np

import rioxarray


import cv2
import matplotlib.pyplot as plt
import rioxarray
from pyproj import Proj, transform


import ultralytics
from ultralytics import YOLO
from ultralytics import settings
#ultralytics.checks()

def utm_to_latlon(utmx, utmy, crs):
    utm_proj = Proj(crs)  # UTM zone 15, northern hemisphere
    lon, lat = utm_proj(utmx, utmy, inverse=True) #x, y
    return lon, lat 

def bbox_coords_to_utm(x_coords, y_coords, xyxy):
    minx, miny, maxx, maxy = xyxy
    # return utm in same order 
    return [x_coords[minx], y_coords[miny], x_coords[maxx], y_coords[maxy]]

def convert_bbox_coords_utm_lat_lon(utm_xyxy, crs):
    #bbox coordinates
    utm_minx, utm_miny, utm_maxx, utm_maxy = utm_xyxy
    lon_lat_minx, lon_lat_miny = utm_to_latlon(utm_minx, utm_miny, crs)
    lon_lat_maxx, lon_lat_maxy = utm_to_latlon(utm_maxx, utm_maxy, crs)
    # return lon/lat in same order 
    return [lon_lat_minx, lon_lat_miny, lon_lat_maxx, lon_lat_maxy]


def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--naip_tile_df", default="/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet", type=str)
    parser.add_argument("--processing_naip_dir", default="/work/csr33/images_for_predictions/processed_naip_data", type=str)
    parser.add_argument("--processing_naip_filename", default="processed_naip_data", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    args = parser.parse_args()
    return args



def main(args):
    #os.chdir("/work/csr33/object_detection")
    #make sure processing naip dir exist
    os.makedirs(args.processing_naip_dir, exist_ok=True)
    #determine chunk-number
    file_name_wo_ext = os.path.splitext(os.path.basename(args.naip_tile_df))[0]
    args.chunk_id = file_name_wo_ext.rsplit("_",1)[1]

    naip_df = pd.read_parquet(args.naip_tile_df) 
    naip_df["img_url"] = naip_df.assets.apply(lambda asset: asset["image"]["href"])#
    data_processing_over_chunk(naip_df, args)
    
    #for df_chunk in chunk_dataframe(naip_df, chunksize=1000):
    #    data_processing_over_chunk(df_chunk, args)
    #    print(chunk)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)