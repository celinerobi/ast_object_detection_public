import os
import argparse
import time

import pystac_client
import planetary_computer
import stackstac

import geopandas as gpd
import numpy as np

import rasterio
from rasterio.merge import merge
from rasterio import mask
from rasterio import plot
import shapely
import ast


def read_raster(item_collection):
    #extract raster data
    raster_paths = [i.assets["data"].href for i in item_collection]
    rasters = [rasterio.open(raster_path) for raster_path in raster_paths]
    assert [raster.nodata == -9999 for raster in rasters]
    return rasters


def ensure_raster_tank_intersect(rasters, tank_geometry):
    raster_geoms = [shapely.box(raster.bounds.left,raster.bounds.bottom, raster.bounds.right, raster.bounds.top) for raster in rasters]
    raster_intersects = [tank_geometry.intersects(raster_geom) for raster_geom in raster_geoms]
    return np.array(rasters)[np.array(raster_intersects)]


def calculate_height(rasters, tank_geometry):
    # tank_geometry : geomtry of tank in utm
    #calculate the height for each raster
    h = []
    w = []
    for raster in rasters:
        clipped_image, clipped_transform = rasterio.mask.mask(raster, [tank_geometry], crop=True)
        #clipped_image.shape
        arr = np.array(clipped_image[clipped_image != -9999])
        if len(arr) > 0:
            h.append(np.quantile(arr.flatten(), 0.95))
            w.append(arr.size)
    [raster.close() for raster in rasters] #close rasters
    # average height
    if len(h) > 0:
        return np.average(h, weights=w)
    else:
        return None

def height_estimation_by_tank(row, catalog, args):
    #create utm geometry
    tank_geometry = shapely.geometry.box(*row["utm_coords"], ccw=True) #utm
    #search catalog using lat lon geometry
    item_collection = catalog.search(collections=[args.collection], 
                            intersects=row.geometry.buffer(0.001)).item_collection()

    if len(item_collection) > 0:
        rasters = read_raster(item_collection)
        # ensure tank the data intersects
        rasters = ensure_raster_tank_intersect(rasters, tank_geometry)
        # calculate height
        height = calculate_height(rasters, tank_geometry)
        return height
    else:
        return None
    

def height_estimation_handle_errors(row, catalog, attempt, args):
    
    try:
        height = height_estimation_by_tank(row, catalog, args)
        return height
    except Exception as err:
        print(type(err))    # the exception type
        time.sleep(args.backoff_factor * (2 ** attempt))
        
        
def height_estimation_retry(row, catalog, args): 
    print(args)
    for attempt in range(args.max_retries):
        try:
            height = height_estimation_handle_errors(row, catalog, attempt, args)
            return height
            print(attempt)
            break
        except:
            if attempt == args.max_retries - 1:
                print(f"Failed after {args.max_retries} attempts!")
            else: 
                wait_time = 2 ** attempt # Exponential backoff
                print(f"Failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)

                
def height_estimation(detected_tanks, args):
    start = time.time()
    height_list = [None] * len(detected_tanks) #height list
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier = planetary_computer.sign_inplace,)
    
    for i, (tank_id, row) in enumerate(detected_tanks.iterrows()):
        #create utm geometry
        height_list[i] = height_estimation_retry(row, catalog, args)
   
    print(time.time() - start)
    return height_list

def get_args_parse():
    parser = argparse.ArgumentParser("Height Estimation")
    parser.add_argument("--prediction_dir", type=str, help="path to the directory storing predictions")
    parser.add_argument("--height_estimation_dir", type=str, help="path to the directory storing height estimation on predictions")
    parser.add_argument("--collection", type=str, help="the name of the planetary computer collection")
    parser.add_argument("--chunk_id",  type=int)
    parser.add_argument("--backoff_factor", default=10, type=float)
    parser.add_argument("--max_retries", default=10, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    detected_tanks = gpd.read_parquet(os.path.join(args.prediction_dir, f"merged_predictions_{args.chunk_id}.parquet"))
    #reformat     
    detected_tanks['utm_coords'] = detected_tanks['utm_coords'].apply(lambda x: ast.literal_eval(x))
    detected_tanks["height"] = height_estimation(detected_tanks, args)
    detected_tanks['utm_coords'] = detected_tanks['utm_coords'].apply(lambda x: str(x))
    
    detected_tanks.to_parquet(os.path.join(args.height_estimation_dir,
                                           f"merged_predictions_height_{args.chunk_id}.parquet"))
    