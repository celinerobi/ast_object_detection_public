"""
Module containing functions to estimation tank volumes
"""

"""
Load Packages
"""


import os
import json
import tempfile
import shutil

import tqdm
from glob import glob

import numpy as np
import pandas as pd
import geopandas as gpd #important

import cv2
import laspy #las open #https://laspy.readthedocs.io/en/latest/
from shapely.ops import transform
from shapely.geometry import Point, Polygon #convert las to gpd
import rioxarray
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
import rtree
#$ pip install pygeos
#import pygeos

import matplotlib as mpl
import matplotlib.pyplot as plt
"""
lidar functions 
"""
def transform_las_points_to_wgs84(las_proj, wgs84_proj, las_x, las_y):
    """ Convert a utm pair into a lat lon pair 
    Args: 
    las_proj(str): the las proj as a proj crs
    las_x(list): a list of the x coordinates for points in las proj
    las_y(list): a list of the y coordinates for points in las proj
    Returns: 
    (Geometry_wgs84): a list of shapely points in wgs84 proj
    """
    #https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    #get utm projection
    Geometry = [Point(xy) for xy in zip(las_x,las_y)] #make x+y coordinates into points
    #transform las into wgs84 point
    project = pyproj.Transformer.from_proj(las_proj, wgs84_proj, always_xy=True).transform
    Geometry_wgs84 = [transform(project, las_point) for las_point in Geometry]
    return(Geometry_wgs84)
"""
DEM Processing
"""
def reproject_dems(initial_dem_dir, final_dem_dir):
    """ Reproject DEMS
    Args: 
    initial_dem_dir(str): directory holding DEMS with original projection
    final_dem_dir(str): directory to hold reprojected imaes 
    """
    initial_dem_paths = glob(initial_dem_dir + "/*.tif")
    dst_crs = 'EPSG:4326'
    for initial_dem_path in initial_dem_paths: #get the bounding box polygons
        dem_name = os.path.splitext(os.path.basename(initial_dem_path))[0]
        with rasterio.open(initial_dem_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': dst_crs,
                            'transform': transform,
                            'width': width,
                            'height': height })
            with rasterio.open(os.path.join(final_dem_dir, dem_name+"_EPSG4326.tif"), 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest)
                    
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

#Get DEM (tif) for each tank
#get utm crs
def get_poly_crs_from_epsg_to_utm(poly):
    """
    Take polygon with coordinates in EPSG get in UTM(meters)
    Args: 
    poly: a shapely olygon objects
    utm_crs: source raster 
    out_img: raster mask
    out_transform: corresponding out transform for the raster mask 
    """
    utm_crs_list = pyproj.database.query_utm_crs_info(datum_name="WGS 84",
                                                      area_of_interest = pyproj.aoi.AreaOfInterest(west_lon_degree=poly.bounds[0],
                                                                                                   south_lat_degree=poly.bounds[1],
                                                                                                   east_lon_degree=poly.bounds[2],
                                                                                                   north_lat_degree=poly.bounds[3],),)
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    return(utm_crs)

def reproject_raster_mask_to_utm(poly, src, clipped_img, clipped_transform, tile_path = None):
    """
    Output values in UTM(meters), coordinates in EPSG
    For a given raster, reproject to a UTM. 
    Args: 
    poly: the original polygon cooresponding to the raster 
    src: source raster 
    out_img: raster mask
    out_transform: corresponding out transform for the raster mask 
    """
    temp_dirpath = tempfile.mkdtemp()
    if tile_path == None:
        tile_path = os.path.join(temp_dirpath, "temp_.tif")
        
    dst_crs = get_poly_crs_from_epsg_to_utm(poly) #utm_crs
    dst_transform, width, height = rasterio.warp.calculate_default_transform(src.crs, dst_crs,
                                                                             src.width, src.height, *src.bounds)
    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
             "height": clipped_img.shape[1],
             "width": clipped_img.shape[2],
             "transform": clipped_transform})

    with rasterio.open(tile_path, 'w', **out_meta) as rast:
        for i in range(1, src.count + 1):
            reproject(source=clipped_img,
                      destination=rasterio.band(rast, i),
                      src_transform=clipped_transform,
                      src_crs=src.crs,
                      dst_transform=dst_transform,
                      dst_crs=dst_crs,
                      resampling=Resampling.nearest)
    #delete temp file
    if os.path.exists(temp_dirpath):
        shutil.rmtree(temp_dirpath)

def get_bounds_for_dems(dem_paths):
    #identify inundation bounds                               
    geometry = []
    dem_names = []
    for dem_path in dem_paths: #get the bounding box polygons
        dem_name = os.path.basename(dem_path)
        dem_names.append(dem_name)
        dem = rasterio.open(dem_path)
        min_lon, min_lat, max_lon, max_lat = dem.bounds
        geometry.append(Polygon([(min_lon,min_lat),(min_lon,max_lat),(max_lon,max_lat),(max_lon,min_lat)]))
    #make dataframe of inundation map bounds
    dem_bounds = gpd.GeoDataFrame({'name': dem_names,'dem_paths': dem_paths,'geometry': geometry})
    return(dem_bounds)

def dem_by_tank(dem_paths, tank_data, output_path):
    #get bounds for dems
    dem_bounds = get_bounds_for_dems(dem_paths)
    dem_bounds.crs = "EPSG:4326"
    #get the dem paths for each tank
    dem_bounds_by_tank = gpd.sjoin(dem_bounds, tank_data, how="right")
    dem_bounds_by_tank = dem_bounds_by_tank.dropna(subset=['dem_paths'])
    
    #iterate over each dem
    tank_poly_grouped_by_dem = dem_bounds_by_tank.groupby(dem_bounds_by_tank.dem_paths) #group gpds by dem
    for dem_path, tank_poly_by_dem in tqdm.tqdm(tank_poly_grouped_by_dem): 
        dem = rasterio.open(dem_path) #load dem
        #iterate over each tank
        tank_poly_by_dem_grouped_by_tank = tank_poly_by_dem.groupby(tank_poly_by_dem.id) #group gpds by tank
        for id_, tank_poly_by_dem_by_tank in tank_poly_by_dem_grouped_by_tank: 
            output_filename = os.path.join(output_path, "DEM_data_tank_id_" + id_ + ".tif") #get output filename
            #get tank polygon
            tank_poly = tank_poly_by_dem_by_tank["geometry"].iloc[0] 
            geo = gpd.GeoDataFrame({'geometry': tank_poly}, index=[0], crs="EPSG:4326")
            coords = getFeatures(geo) 
            #clip dem to tank
            clipped_img, clipped_transform = rasterio.mask.mask(dataset=dem, shapes=coords, crop=True)
            #reproject
            reproject_raster_mask_to_utm(tank_poly, dem, clipped_img, clipped_transform, output_filename)
            
def identify_tank_ids(lidar_by_tank_output_path, DEM_by_tank_output_path):
    regex = re.compile(r'\d+')
    #the tank ids with corresponding lidar data 
    tank_ids_lidar = []
    lidar_by_tank_geojson_name = os.listdir(lidar_by_tank_output_path)
    for lidar_by_tank in lidar_by_tank_geojson_name:
        tank_ids_lidar.append([int(x) for x in regex.findall(lidar_by_tank)][0])

    #the tank ids with corresponding DEM data 
    #vol_est.remove_thumbs(DEM_by_tank_output_path) #remove thumbs 
    tank_ids_dem = []
    DEM_by_tank_tif_name = os.listdir(DEM_by_tank_output_path)
    for DEM_by_tank in DEM_by_tank_tif_name:
        tank_ids_dem.append([int(x) for x in regex.findall(DEM_by_tank)][0])

    #the tank ids with both lidar and DEMs
    tank_ids = list(set(tank_ids_dem).intersection(tank_ids_lidar))
    tank_ids = [str(i) for i in tank_ids]

    #paths to the DEM and lidar data 
    DEM_path_for_height = []
    lidar_path_for_height = []

    for tank_id in tank_ids:
        lidar_path_by_tank_for_height.append(os.path.join(lidar_by_tank_output_path, [string for string in lidar_by_tank_geojson_name if tank_id in string][0]))
        DEM_path_by_tank_for_height.append(os.path.join(DEM_by_tank_output_path, [string for string in DEM_by_tank_tif_name if tank_id in string][0]))
    return(lidar_path_by_tank_for_height, DEM_path_by_tank_for_height)

def add_bare_earth_data_to_lpc_by_tank_data(lidar_path_by_tank_for_height, DEM_path_by_tank_for_height):
    for i, (lidar_path, DEM_path) in enumerate(zip(lidar_path_by_tank_for_height, DEM_path_by_tank_for_height)):
        # Read in each lidar dataset
        lidar = gpd.read_file(lidar_path)
        lidar_coords = [(x,y) for x, y in zip(lidar["X coordinate"], lidar["Y coordinate"])] #'EPSG:4326' coords

        # Open the DEM raster data and store metadata
        dem_src = rasterio.open(DEM_path)

        # Sample the raster at every point location and store values in DataFrame
        #pts['Raster Value'] = [x for x in src.sample(coords)]
        #pts['Raster Value'] = probes.apply(lambda x: x['Raster Value'][0], axis=1)
        lidar['bare_earth_elevation'] = [z[0] for z in dem_src.sample(coords)]

        with open(lidar_path, "w") as file:
            file.write(lidar.to_json()) 

"""
Add average base elevation to dataframe
"""     
def average_bare_earth_elevation_for_tanks(gdf, tank_data, dem_paths):    
    """ Calculate the diameter of a given bounding bbox for imagery of a given resolution
    Arg:
    bbox(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 
    resolution(float): the (gsd) resolution of the imagery
    Returns:
    (diameter): the diameter of the bbox of interest
    """

    tank_data_w_lpc = gpd.sjoin(dem_bounds, tank_data, how = "left")

    for tank_index, tank_poly in tqdm.tqdm(enumerate(tank_data["geometry"])): #iterate over the tank polygons
        if dem_bounds.contains(tank_poly): #identify whether the tank bbox is inside of the state polygon
            index.append(tank_index) #add state name for each tank to list 
    tank_data_in_lidar_extent = tank_data.iloc[index]
    """
    #get average bare earth elevation values values
    for dem_index, dem_poly in enumerate(dem_bounds["geometry"]): #iterate over the dem polygons
        if dem_poly.contains(tank_poly): #identify whether the bbox is inside of the dem map
            #make a geodataframe for each tank polygon that is contained within the dem
            geo = gpd.GeoDataFrame({'geometry': tank_poly}, index=[0], crs=gdf.crs)
            coords = getFeatures(geo) 
            dem = rasterio.open(dem_paths[dem_index])
            out_img, out_transform = rasterio.mask.mask(dataset=dem, shapes=coords, crop = True)
            #average_bare_earth_elevation[tank_index] = np.average(out_img)
            #average to reprojected raster
            out_img_utm = reproject_raster_mask_to_utm(tank_poly, dem, out_img, out_transform)
            average_bare_earth_elevation[tank_index] = np.average(out_img_utm)
    #add inundation values to tank database 
    return(gdf)

    #3. Get the extent of the Lidar data 
    minx, miny, maxx, maxy = lidar["geometry"].total_bounds
    lidar_extent = Polygon([(minx,miny), (minx,maxy), (maxx,maxy), (maxx,miny)])
    

    
    #5. Get the LP corresponding with the tank dataset
    tank_data_w_lpc = gpd.sjoin(tank_data_in_lidar_extent,lidar)
    tank_data_w_lpc = tank_data_w_lpc.dropna(subset=['Z coordinate'])
    #save geodatabase as json
    with open(os.path.join(args.output_tile_level_annotation_path, las_name+".geojson"), 'w') as file:
        file.write(tank_data_w_lpc.to_json()) 
    """
    
def remove_thumbs(path_to_folder_containing_images):
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_images(str): path to folder containing images
    Returns:
    None
    """
    if len(glob(path_to_folder_containing_images + "/*.db", recursive = True)) > 0:
        os.remove(glob(path_to_folder_containing_images + "/*.db", recursive = True)[0])