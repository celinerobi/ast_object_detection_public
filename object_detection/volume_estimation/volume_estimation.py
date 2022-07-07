"""
Correct inconsistent labels
"""

"""
Import Packages
"""
import tqdm
import numpy as np
import pandas as pd
import os
import geopandas as gpd #important
import cv2
import laspy #las open #https://laspy.readthedocs.io/en/latest/
from shapely.ops import transform
from shapely.geometry import Point,Polygon #convert las to gpd
import rioxarray
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
from pyproj import CRS
import rtree
import argparse
def get_args_parse():
    parser = argparse.ArgumentParser(description='This script adds LPC data to tile level tank data')
    parser.add_argument('--lidar_path', type=str, default=None,
                        help='path to lidar tile (laz file)')
    parser.add_argument('--tile_level_annotation_path', type=str, default=None, 
                        help='tile level tank annotations')
    parser.add_argument('--output_tile_level_annotation_path', type=str, default=None, 
                        help='output tile level tank annotations')
    args = parser.parse_args()
    return args

def main(args):
    #1. Read in lidar data using laspy 
    las_name = os.path.splitext(os.path.basename(args.lidar_path))[0]
    las = laspy.read(args.lidar_path)
    las_crs = pyproj.CRS.from_user_input(las.header.vlrs.get("WktCoordinateSystemVlr")[0].string) #get crs
    
    #2. CONVERTING LAS TO PANDAS#
    #Import LAS into numpy array (X=raw integer value; x=scaled float value)
    lidar_points = np.array((las.x, las.y, las.z,las.intensity)).transpose()
    #Transform to pandas DataFrame
    lidar_df = pd.DataFrame(lidar_points, columns = ['X coordinate', 'Y coordinate', 'Z coordinate', 'Intensity'])
    # transform into the geographic coordinate system
    Geometry = [Point(xy) for xy in zip(las.x,las.y)] #make x+y coordinates into points
    EPSG4326 = pyproj.CRS('EPSG:4326')
    project = pyproj.Transformer.from_proj(las_crs, EPSG4326).transform
    Geometry_gcs = [transform(project, las_point) for las_point in Geometry]
    #Transform to geopandas GeoDataFrame
    lidar = gpd.GeoDataFrame(lidar_df, crs = EPSG4326, geometry=Geometry_gcs) #set correct spatial reference
    lidar = lidar.to_crs('EPSG:4326')
    
    #3. Get the extent of the Lidar data 
    miny, minx, maxy, maxx = lidar["geometry"].total_bounds
    lidar_extent = Polygon([(minx,miny), (minx,maxy), (maxx,maxy), (maxx,miny)])
    
    #4. Subset the tank dataset to the lidar data
    tank_data = gpd.read_file(args.tile_level_annotation_path)
    tank_data = tank_data.to_crs('EPSG:4326')
    tank_data.crs = "EPSG:4326" #assign projection

    index = []
    for tank_index, tank_poly in tqdm.tqdm(enumerate(tank_data["geometry"])): #iterate over the tank polygons
        if lidar_extent.contains(tank_poly): #identify whether the tank bbox is inside of the state polygon
            index.append(tank_index) #add state name for each tank to list 
    tank_data_in_lidar_extent = tank_data.iloc[index]
    #Get the LP corresponding with the tank dataset
    tank_data_w_lpc = gpd.sjoin(tank_data_in_lidar_extent,lidar, how='left', predicate='contains')
    tank_data_w_lpc = tank_data_w_lpc.dropna(subset=['Z coordinate'])

    #save geodatabase as json
    with open(os.path.join(args.output_tile_level_annotation_path, las_name+".geojson"), 'w') as file:
        file.write(tank_data_w_lpc.to_json()) 
        
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
