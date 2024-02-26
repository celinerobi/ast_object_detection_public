import os
import argparse
import math
import time
import ast
from copy import copy
from glob import glob

import pandas as pd
import numpy as np
import pyproj 
from pyproj import Proj
import shapely
import geopandas as gpd
from shapely.ops import transform
from shapely.geometry import Point
import rioxarray

import torch

import ultralytics
from ultralytics import YOLO

 
def chunk_df(img_paths, num_chunks=None, rows_per_chunk=None):
    # Calculate the number of rows per chunk
    if num_chunks is None and rows_per_chunk is None:
        print("no chunkable value is given")
    
    if num_chunks is not None and rows_per_chunk is not None:
        print("to many chunkable value is given")
    
    if num_chunks is not None:
        rows_per_chunk = len(img_paths) // num_chunks
    
    df_chunks = [np.array(img_paths[i : i + rows_per_chunk]) for i in range(0, len(img_paths), rows_per_chunk)]
    return df_chunks


def tile_dimensions_and_utm_coords(tile_path): #used
    """ Obtain tile band, height and width and utm coordinates
    Args: tile_path(str): the path of the tile 
    Returns: 
    utmx(np array): the x utm coordinates corresponding with the tile coordinate convention (origin in upper left hand corner)
    utmy(np array): the y utm coordinates corresponding with the tile coordinate convention (origin in upper left hand corner)
    tile_band(int): the number of bands
    tile_height(int): the height of the tile (in pixels)
    tile_width(int): the width of the tile (in pixels)
    """
    ## Get tile locations
    da = rioxarray.open_rasterio(tile_path) ## Read the data
    # Compute the lon/lat coordinates with rasterio.warp.transform
    # lons, lats = np.meshgrid(da['x'], da['y'])
    tile_band, tile_height, tile_width = da.shape[0], da.shape[1], da.shape[2]
    utmx = np.array(da['x'])
    utmy = np.array(da['y'])
    crs =  str(rioxarray.open_rasterio(tile_path).rio.crs)
    return(utmx, utmy, crs, tile_band, tile_height, tile_width)
    del da


def process_results(results, tile_height, tile_width, item_dim):
    #xyxys = []
    bbox_pixel_coords_list = [] #xyxy coords with repsect to the tile
    conf_list = [] #probability
    class_name_list = [] #class name
    image_names_list = []
    tile_names_list = []
    lat_lons = []
    for result in results:
        boxes = result.boxes
        image_name = os.path.splitext(os.path.basename(result.path))[0]
        tile_name = image_name.rsplit("_",2)[0]
        if len(boxes) > 0: 
            #get valeus for all bounding boxes
            #class name
            class_name_list.extend([result.names[class_number] for class_number in boxes.cls.cpu().detach().tolist()])
            
            conf_list.extend(boxes.conf.cpu().detach().tolist())

            xyxy = boxes.xyxy.cpu().detach().numpy() - 1 #read xmin,ymin,xmax,ymax coordinates to memory as a numpy array
            xyxy = np.round(xyxy).astype(np.int32).tolist()  # round so that it can be used for utm to lonlat conversion, check if zero indexed
            image_names_list.extend([image_name]*len(xyxy)) # The index is a six-digit number like '000023'.
            tile_names_list.extend([tile_name]*len(xyxy)) # The index is a six-digit number like '000023'.
            #calculate the tile level pixel coordinates
            bbox_pixel_coords_list.extend([calculate_tile_level_bbox(image_name, box, item_dim,
                                                                     tile_width, tile_height) for box in xyxy])
        del boxes
    return pd.DataFrame({"confidence":conf_list, "class_name": class_name_list,
                       "image_names": image_names_list, "tile_names": tile_names_list, 
                         "bbox_pixel_coords": bbox_pixel_coords_list})#,  dtype=dtypes


def predict_process(img_paths, tile_height, tile_width, model, args):
    # obtain predictions over the dataframe
    results_df = pd.DataFrame({})
    for df_chunk in chunk_df(img_paths, rows_per_chunk=50):
        results = model.predict(df_chunk.tolist(), save=False, imgsz=args.imgsz)#, conf=0.5)
        #process_results(results, utmx, utmy, utm_proj, tile_height, tile_width, item_dim=args.imgsz)
        results_df = pd.concat([results_df, process_results(results, tile_height, tile_width, item_dim=args.imgsz)])
        del results, df_chunk
    return results_df


def calculate_tile_level_bbox(image_name, xyxy, item_dim, tile_width, tile_height):
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = xyxy
    #identify rows and columns
    y, x = image_name.split("_")[-2:] #name of tif with the extension removed; y=row;x=col
    # Each chip xml goes from 1 - item_dim, specify the "0", or the end point of the last xml
    image_minx = int(x)*item_dim
    image_miny = int(y)*item_dim

    #add the bounding boxes
    obj_xmin = image_minx + obj_xmin
    obj_ymin = image_miny + obj_ymin
    obj_xmax = image_minx + obj_xmax
    obj_ymax = image_miny + obj_ymax
    
    # correct bboxes that extend past the bounds of the tile width/height
    if int(obj_xmin) >= tile_width:
        obj_xmin = tile_width - 1
    if int(obj_xmax) >= tile_width:
        obj_xmax = tile_width - 1
    if int(obj_ymin) >= tile_height:
        obj_ymin = tile_height - 1
    if int(obj_ymax) >= tile_height:
        obj_ymax = tile_height - 1
    
    return [obj_xmin, obj_ymin, obj_xmax, obj_ymax]


def transform_point_utm_to_wgs84(utm_proj, utm_xcoord, utm_ycoord): #used
    """ Convert a utm pair into a lat lon pair 
    Args: 
    utm_proj(str): the UTM string as the in term of EPSG code
    utmx(int): the x utm coordinate of a point
    utmy(int): the y utm coordinates of a point
    Returns: 
    (wgs84_pt.x, wgs84_pt.y): the 'EPSG:4326' x and y coordinates 
    """
    #https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    #get utm projection
    utm = pyproj.CRS(utm_proj)
    #get wgs84 proj
    wgs84 = pyproj.CRS('EPSG:4326')
    #specify utm point
    utm_pt = Point(utm_xcoord, utm_ycoord)
    #transform utm into wgs84 point
    project = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    wgs84_pt = transform(project, utm_pt)
    return wgs84_pt.x, wgs84_pt.y
    
    
def get_utm_coords(pixel_coords, utmx, utmy):
    minx, miny, maxx, maxy = pixel_coords
    return [utmx[minx], utmy[miny], utmx[maxx], utmy[maxy]]  
    
    
def get_lat_lon_coords(utm_coords, utm_proj):
    minx, miny, maxx, maxy = utm_coords
    #determine the lat/lon
    nw_lon, nw_lat = transform_point_utm_to_wgs84(utm_proj, minx, miny)
    se_lon, se_lat = transform_point_utm_to_wgs84(utm_proj, maxx, maxy) 
    return [nw_lon, nw_lat, se_lon, se_lat]

        
def merge_boxes(bbox1, bbox2): #used
    """ 
    Generate a bounding box that covers two bounding boxes
    Called in merge_algo
    Arg:
    bbox1(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 1 
    bbox2(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 2
    Returns:
    merged_bbox(list): a list of the (xmin, ymin, xmax, ymax) coordinates for the merged bbox

    """
    return [min(bbox1[0], bbox2[0]), 
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])]


def calc_sim(bbox1, bbox2, dist_limit): #used
    """Determine the similarity of distances between bboxes to determine whether bboxes should be merged
    Computer a Matrix similarity of distances of the text and object
    Called in merge_algo
    Arg:
    bbox1(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 1 
    bbox2(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 2
    dist_list(int): the maximum threshold (pixel distance) to merge bounding boxes
    Returns:
    (bool): to indicate whether the bboxes should be merged 
    """

    # text: ymin, xmin, ymax, xmax
    # obj: ymin, xmin, ymax, xmax
    bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax = bbox1
    bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax = bbox2
    x_dist = min(abs(bbox2_xmin-bbox1_xmax), abs(bbox2_xmax-bbox1_xmin))
    y_dist = min(abs(bbox2_ymin-bbox1_ymax), abs(bbox2_ymax-bbox1_ymin))
        
    #define distance if one object is inside the other
    if (bbox2_xmin <= bbox1_xmin) and (bbox2_ymin <= bbox1_ymin) and (bbox2_xmax >= bbox1_xmax) and (bbox2_ymax >= bbox1_ymax):
        return True
    elif (bbox1_xmin <= bbox2_xmin) and (bbox1_ymin <= bbox2_ymin) and (bbox1_xmax >= bbox2_xmax) and (bbox1_ymax >= bbox2_ymax):
        return True
    #determine if both bboxes are close to each other in 1d, and equal or smaller length in the other
    elif (x_dist <= dist_limit) and (bbox1_ymin <= bbox2_ymin) and (bbox1_ymax >= bbox2_ymax): #bb1 bigger
        return True
    elif (x_dist <= dist_limit) and (bbox2_ymin <= bbox1_ymin) and (bbox2_ymax >= bbox1_ymax): #bb2 bigger
        return True
    elif (y_dist <= dist_limit) and (bbox1_xmin <= bbox2_xmin) and (bbox1_xmax >= bbox2_xmax): #bb1 bigger
        return True
    elif (y_dist <= dist_limit) and (bbox2_xmin <= bbox1_xmin) and (bbox2_xmax >= bbox1_xmax): #bb2 bigger
        return True
    else: 
        return False

    
def merge_predicted_bboxes(results_df, dist_limit = 5):
    class_names = results_df.class_name.to_list()
    bbox_pixel_coords = results_df.bbox_pixel_coords.to_list()
    confidences = results_df.confidence.to_list()
    tile_names = results_df.tile_names.to_list()
    merge_bools = [False] * len(class_names)
    for i, (conf1, class_name1, bbox1, tile_name1) in enumerate(zip(confidences, class_names, bbox_pixel_coords, tile_names)):
        for j, (conf2, class_name2, bbox2, tile_name2) in enumerate(zip(confidences, class_names, bbox_pixel_coords, tile_names)):
            if j <= i: #only consider the remaining bboxes
                continue
            # Create a new box if a distances is less than distance limit defined 
            merge_bool = calc_sim(bbox1, bbox2, dist_limit) 
            if merge_bool == True:
                # Create a new box  
                new_box = merge_boxes(bbox1, bbox2)   
                bbox_pixel_coords[i] = new_box
                #delete previous text boxes
                del bbox_pixel_coords[j]
                class_name_merge = [class_name1, class_name2]
                conf = [conf1, conf2]
                #determine the class with the highest conf
                max_conf_idx = np.argmax(conf)
                max_conf_class = class_name_merge[max_conf_idx]


                class_names[i] = max_conf_class
                confidences[i] = np.mean(conf)

                conf
                #delete previous text 
                del class_names[j],  confidences[j], tile_names[j]
    return pd.DataFrame({"confidence":confidences, "class_name": class_names,
                         "bbox_pixel_coords": bbox_pixel_coords, "tile_names": tile_names})#,  dtype=dtypes


def write(predictions, predictions_file_path):
    # remove file if it exists 
    if os.path.exists(predictions_file_path):
        predictions.to_parquet(predictions_file_path, engine='fastparquet', append=True)
    else:
        predictions.to_parquet(predictions_file_path, engine='fastparquet')
        

def calculate_diameter(bbox, resolution = 0.6): #used
    """ Calculate the diameter of a given bounding bbox (in Pascal Voc Format) for imagery of a given resolution
    Arg:
    bbox(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box. Utm coordinates are provided as [nw_x_utm, se_y_utm, se_x_utm, nw_y_utm] to conform with Pascal Voc Format.
    resolution(float): the (gsd) resolution of the imagery
    Returns:
    (diameter): the diameter of the bbox of interest
    """
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = bbox
    obj_width = obj_xmax - obj_xmin
    obj_height = obj_ymin - obj_ymax 
    diameter = min(obj_width, obj_height) * resolution #meter
    return diameter

        
def get_args_parse():
    parser = argparse.ArgumentParser("Predict on images")    
    parser.add_argument("--chunk_id",  type=int)
    parser.add_argument("--tile_dir", default="/work/csr33/images_for_predictions/naip_tiles", type=str)
    parser.add_argument("--tilename_chunks_path", default='/hpc/home/csr33/ast_object_detection/images_for_prediction/tilename_chunks.npz', type=str)
    parser.add_argument("--model_path", default="/work/csr33/object_detection/runs/detect/baseline_train/weights/best.pt", type=str)
    parser.add_argument("--prediction_dir", default="/work/csr33/images_for_predictions/predictions", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument('--img_dir', type=str, default="/work/csr33/images_for_predictions/naip_imgs")
    parser.add_argument('--classification_threshold', type=float, default=0.5)

    args = parser.parse_args()
    return args

def predict(args):
    torch.cuda.set_device(0) # Set to your desired GPU number
    #determine chunk-number   
    os.makedirs(args.prediction_dir, exist_ok=True)
    model = YOLO(args.model_path)  # custom trained model 
    
    # load a subset of the tile paths to predict on
    tile_paths = np.load(args.tilename_chunks_path)[str(args.chunk_id)]
    tile_names = [os.path.splitext(os.path.basename(tile_path))[0] for tile_path in tile_paths]
    
    #intialize dataframes
    predict_df = pd.DataFrame({})
    merged_df = pd.DataFrame({})
    
    # obtain predictions over the dataframe
    for tile_name in tile_names:
        print("tile_name", tile_name)
        start_time = time.time()
        img_paths = glob(os.path.join(args.img_dir,"*"+tile_name+"*")) #identify the imgs correspondig to a given tile
        tile_path = os.path.join(args.tile_dir, tile_name +".tif") # specify the tile path
        #obtain tile information
        utmx, utmy, utm_proj, tile_band, tile_height, tile_width = tile_dimensions_and_utm_coords(tile_path) #used
        #predict on images
        predict_df_by_tank = predict_process(img_paths, tile_height, tile_width, model, args)
        predict_df_by_tank = copy(predict_df_by_tank[predict_df_by_tank.confidence > args.classification_threshold])
        #merge neighboring images
        merged_df_by_tank = merge_predicted_bboxes(predict_df_by_tank, dist_limit = 5)
        # calculate utm and lat lon coords
        
        merged_df_by_tank["utm_coords"] = merged_df_by_tank["bbox_pixel_coords"].apply(\
                                             lambda pixel_coords: get_utm_coords(pixel_coords, utmx, utmy))
        merged_df_by_tank["latlon_coords"] = merged_df_by_tank["utm_coords"].apply(\
                                             lambda utm_coords: get_lat_lon_coords(utm_coords, utm_proj))
        merged_df_by_tank["diameter"] = merged_df_by_tank["utm_coords"].apply(lambda utm_coord: calculate_diameter(utm_coord, resolution = 1))
        #specify the projection used 
        merged_df_by_tank["utm_proj"] = [utm_proj] * len(merged_df_by_tank)
      #update dataframes
        predict_df = pd.concat([predict_df, predict_df_by_tank], ignore_index=True)
        merged_df = pd.concat([merged_df, merged_df_by_tank], ignore_index=True)
        #delete temp dataframe to conserve memory
        del predict_df_by_tank, merged_df_by_tank
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")   
        
    predict_df.to_parquet(os.path.join(args.prediction_dir, f"predictions_{args.chunk_id}.parquet"))

    #reformat lat lon to geometry
    merged_df["geometry"] = merged_df["latlon_coords"].apply(lambda latlon_coord: 
                                                             shapely.geometry.box(*latlon_coord))
    
    merged_df.drop(columns='latlon_coords', inplace=True)
    merged_df['utm_coords'] = merged_df['utm_coords'].apply(lambda x: str(x))
    merged_df['class_name'] = merged_df['class_name'].apply(lambda x: str(x))
    merged_df['bbox_pixel_coords'] = merged_df['bbox_pixel_coords'].apply(lambda x: str(x))
    merged_df['confidence'] = merged_df['confidence'].apply(lambda x: str(x))


    merged_df = gpd.GeoDataFrame(merged_df)
    merged_df.to_parquet(os.path.join(args.prediction_dir, f"merged_predictions_{args.chunk_id}.parquet"))


if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    predict(args)