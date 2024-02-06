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


def reformat_data_chunk(df_chunk, args):
    df_chunk["y_coords"] = df_chunk["y_coords"].apply(bytestring_to_array)
    df_chunk["x_coords"] = df_chunk["x_coords"].apply(bytestring_to_array)
    df_chunk["split_tile_values"] = df_chunk["split_tile_values"].apply(bytestring_to_array)
    df_chunk["split_tile_values"] = df_chunk["split_tile_values"].apply(lambda v: v.reshape((args.imgsz, args.imgsz, 3)))
    return df_chunk


def bytestring_to_array(bytestring):
    if type(bytestring) == bytes:
        decoded_string = bytestring.decode("utf-8")
        # Use ast.literal_eval to safely evaluate the string to a Python object
        list_obj = np.array(ast.literal_eval(decoded_string))
        return list_obj
    else: 
        return bytestring
    
    
def run_prediction(model, df_chunk):
    #predict
    conf_list = []
    class_list = []
    utm_bboxes_list = []
    lon_lat_bbox_list = []
    split_tile_values = [s.astype(np.uint8) for s in df_chunk["split_tile_values"]] #remove after preidct is run again
    results = model.predict(split_tile_values, save=False, imgsz=args.imgsz)#, conf=0.5)
    # process predictions
    for result, x_coords, y_coords, crs in zip(results, df_chunk["x_coords"].to_list(), 
                                               df_chunk["y_coords"].to_list(), 
                                               df_chunk["crs"].to_list()):
        boxes = result.boxes
        if len(boxes) > 0:             
            xyxys = boxes.xyxy.cpu().detach().numpy() #read xmin,ymin,xmax,ymax coordinates to memory as a numpy array
            xyxys = np.round(xyxys).astype(np.int32)  - 1 # round so that it can be used for utm to lonlat conversion, check if zero indexed
            utm_bboxes = [bbox_coords_to_utm(x_coords, y_coords, xyxy) for xyxy in xyxys]
            utm_bboxes_list.extend(utm_bboxes)
            
            lon_lat_bbox = [convert_bbox_coords_utm_lat_lon(utm_bbox, crs) for utm_bbox in utm_bboxes]
            lon_lat_bbox_list.extend(lon_lat_bbox)
    
            class_list.extend(boxes.cls.cpu().detach().tolist())
            conf_list.extend(boxes.conf.cpu().detach().tolist())
            
    return pd.DataFrame({"class": class_list, "confidence":conf_list, "utm_bboxes": utm_bboxes_list,
                         "lon_lat_bboxes": lon_lat_bbox_list})#,  dtype=dtypes)


def chunk_dataframe(df, chunksize):
    """
    Split dataframe into chunks of specified size.
    Args:
        df (pandas.DataFrame): DataFrame to chunk 
        chunksize (int): Size of each chunk
    Returns:
        generator: A generator yielding the chunked dataframes
    """

    for i in range(0, len(df), chunksize):
        yield df.iloc[i:i + chunksize]
        
        
def get_args_parse():
    parser = argparse.ArgumentParser("")    
    parser.add_argument("--processing_naip_dir", default="/work/csr33/images_for_predictions/processed_naip_data", type=str)
    parser.add_argument("--processing_naip_filename", default="processed_naip_data", type=str)
    parser.add_argument("--chunk_id",  type=int)

    parser.add_argument("--model_path", default="/work/csr33/object_detection/runs/detect/baseline_train/weights/best.pt", type=str)
    parser.add_argument("--prediction_dir", default="/work/csr33/images_for_predictions/predictions", type=str)
    parser.add_argument("--prediction_filename", default="predictions", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    args = parser.parse_args()
    return args


def predict(args):
    processed_naip_file_path = os.path.join(args.processing_naip_dir, f"{args.processing_naip_filename}_{args.chunk_id}.parquet")
    processed_naip_df = pd.read_parquet(processed_naip_file_path) 
    #os.chdir("/work/csr33/object_detection")
    #make sure processing naip dir exist
    #determine chunk-number    
    os.makedirs(args.prediction_dir, exist_ok=True)
    predictions_file_path = os.path.join(args.prediction_dir, f"{args.prediction_filename}_{args.chunk_id}.parquet")

    model = YOLO(args.model_path)  # custom trained model 
    # obtain predictions over the dataframe
    predictions_df = pd.DataFrame()
    for df_chunk in chunk_dataframe(processed_naip_df, chunksize=50):
        #print(df_chunk)
        df_chunk = reformat_data_chunk(copy(df_chunk), args)
        predictions_df = pd.concat([predictions_df, run_prediction(model, df_chunk)])
    predictions_df.to_parquet(predictions_file_path, engine='fastparquet')


if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)