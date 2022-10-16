"""
Module containing functions to estimation tank volumes
"""
##########################################################################################################################
############################################          Load Packages         ##############################################
##########################################################################################################################
import os
import sys
import json
import tempfile
import shutil
import re
import copy
from glob import glob

import tqdm
#import rtree

import urllib3
import requests

import numpy as np
import pandas as pd
import geopandas as gpd #important

import laspy #las open #https://laspy.readthedocs.io/en/latest/
from shapely.ops import transform
from shapely.geometry import Point, Polygon #convert las to gpd
import rioxarray
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
#import pygeos

import cv2
#import torch
#import fastai
from skimage import data, filters, exposure, measure, segmentation, morphology, color

import matplotlib as mpl
mpl.rc('image', cmap='gray')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar # For dealing with Colorbars the proper way - TBD in a separate PyCoffee ?
##########################################################################################################################
############# General Functions: Remove when package is pushed (in form_calcs)     #################################################
##########################################################################################################################
def remove_thumbs(path_to_folder_containing_images):
    """ Remove Thumbs.db file from a given folder
    Args: 
    path_to_folder_containing_images(str): path to folder containing images
    Returns:
    None
    """
    if len(glob(path_to_folder_containing_images + "/*.db", recursive = True)) > 0:
        os.remove(glob(path_to_folder_containing_images + "/*.db", recursive = True)[0])
        
## Write files
def write_list(list_, file_path):
    print("Started writing list data into a json file")
    with open(file_path, "w") as fp:
        json.dump(list_, fp)
        print("Done writing JSON data into .json file")

# Read list to memory
def read_list(file_path):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        list_ = json.load(fp)
        return list_

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

##########################################################################################################################
################################################  USGS TNM API Functions #################################################
##########################################################################################################################
def submit_http_api_request(payload, url):
    """ Use get http request to pull information from api
    Args:
        payload dict): specify components to use to pull from
        url(str): the url corresponding to the api 
    Returns:
        (bool,contents_json): returns a True/False (True if the request was sucessful, False otherwise); 
        If true a json of the file of interest is returned, otherwise False is returned.
    """
    response = requests.get(url, payload)
    #print(response.url)
    # We can check whether the request is sucessful, (200 -> OK) or nother errors (i.e, 404 -> Not Found)
    if response.status_code  == 200:
        response.encoding = 'utf-8' # Optional: requests infers this internally
        contents_json = response.json()
        return(True, contents_json)
    else:         
        #print("An error occurred", response.status_code)
        return(False, False)

def verify_entries_exist(contents_json, entries_idx):
    """ Determine whether of not the pull request returned any entries
    Args: 
        contents_json(json): the response from the get request 
        entries_idx(str): key indiciating the enteries of interest
    Returns:
        bool: True if there are enteries, False if there are not enteries 
    """
    if int(contents_json[entries_idx]) == 0:
        return(False)
    else:
        return(True)

def json_to_dict(json, json_idx, nested_idx):
    """Convert json file storing nested dictionaries to a dicitionary using a given index. Save dictionary as pandas df. 
    Args:
        json(json):
        json_idx(str):
        nested_idx(str): nested idx in json used as key value for new dictionary

    return:
        df: dataframe of restructured json file
    """
    dict_ = {}
    for item in json[json_idx]:
        col_idx = item.keys()
        title = item[nested_idx]
        dict_[title] = item
    df = pd.DataFrame(dict_,index=col_idx)
    df = df.T
    return(df)

def get_dataset_of_interest(contents_df, name_list, url_list, geometry_list):
    newest = contents_df.loc[contents_df["dateCreated"] == contents_df["dateCreated"].max()]
    url = newest['downloadLazURL'][0]
    bbox = newest['boundingBox'][0]
    polygon = Polygon([(bbox["minX"],bbox["minY"]), (bbox["minX"],bbox["maxY"]), 
                         (bbox["maxX"],bbox["maxY"]), (bbox["maxX"],bbox["minY"])])
    #add to list
    name_list.append(newest.index[0])
    url_list.append(url)
    geometry_list.append(polygon)
    return(name_list, url_list, geometry_list)

def usgs_api(tile_level_annotations, tnm_url, dataset_name, request_total_idx, request_content_idx, request_content_names_idx):
    #create lists to store data 
    tnm_names = []
    urls = []
    geometries = []
    tank_idxs = []

    #iterate over tanks
    for tank_idx, row in tqdm.tqdm(tile_level_annotations.iterrows(), total=tile_level_annotations.shape[0]):
        minx = row['nw_corner_polygon_lon']
        miny = row['se_corner_polygon_lat']
        maxx = row['se_corner_polygon_lon']
        maxy = row['nw_corner_polygon_lat']
        tank_polygon = Polygon([(minx,miny), (minx,maxy), (maxx,maxy), (maxx,miny)])
        tank_bbox = [minx, miny, maxx, maxy]
        tank_bbox_str = ",".join(map(str, tank_bbox)) #https://stackoverflow.com/questions/32313623/passing-comma-separated-values-in-request-get-python

        tnm_names_itr = copy.copy(tnm_names)
        urls_itr = copy.copy(urls)
        geometries_itr = copy.copy(geometries)

        #check if tank polygon is inside of a previously identified dataset
        ## https://stackoverflow.com/questions/4406389/if-else-in-a-list-comprehension
        matching_poly= [(tnm_name, url, tnm_poly) for tnm_name, url, tnm_poly in zip(tnm_names_itr, urls_itr, geometries_itr) if tnm_poly.contains(tank_polygon)]
        if len(matching_poly) > 0: #if it is inside of another dataset, add to the lists tracking names, urls, geoms, idx
            tnm_name, url, tnm_poly = matching_poly[0]
            tnm_names.append(tnm_name)
            urls.append(url)
            geometries.append(tnm_poly)
            tank_idxs.append(tank_idx)
        else:
            payload = {'bbox': tank_bbox_str, "datasets": dataset_name}
            bool_request, contents_json = submit_http_api_request(payload, tnm_url)
            if bool_request: #if the request was successful
                bool_entries = verify_entries_exist(contents_json, request_total_idx)
                if bool_entries: #if the request provided enteries
                    contents_df = json_to_dict(contents_json, request_content_idx, request_content_names_idx) 
                    tnm_names, urls, geometries = get_dataset_of_interest(contents_df, tnm_names, urls, geometries)
                    tank_idxs.append(tank_idx)
    
    complete_df = pd.DataFrame(list(zip(tank_idxs, tnm_names, urls, geometries)), 
                               index = tank_idxs, columns =['tank_idxs','usgs_tnm_names', 'urls', 'geometry'])
    return(complete_df)
##########################################################################################################################
############################################   Lidar Processing Functions   ##############################################
##########################################################################################################################
def project_list_of_points(initial_proj, final_proj, x_points, y_points):
    """ Convert a utm pair into a lat lon pair 
    Args: 
    initial_proj(str): the initial as a (proj crs)
    x_points(list): a list of the x coordinates for points to project
    y_points(list): a list of the y coordinates for points in las proj
    Returns: 
    (Geometry_wgs84): a list of shapely points in wgs84 proj
    """
    #https://gis.stackexchange.com/questions/127427/transforming-shapely-polygon-and-multipolygon-objects
    #get utm projection
    Geometry = [Point(xy) for xy in zip(x_points,y_points)] #make x+y coordinates into points
    #transform las into wgs84 point
    project = pyproj.Transformer.from_proj(initial_proj, final_proj, always_xy=True).transform
    Geometry_projected = [transform(project, point) for point in Geometry]
    return(Geometry_projected)

##########################################################################################################################
############################################    DEM Processing Functions    ##############################################
##########################################################################################################################
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

#Get DEM (tif) for each tank
#get utm crs
def get_poly_crs_from_epsg_to_utm(poly):
    """
    Take polygon with coordinates in EPSG get in UTM(meters)
    Args: 
    poly: a shapely olygon objects
    Returns:
    utm_crs: source raster 
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
    #remove_thumbs(DEM_by_tank_output_path) #remove thumbs 
    tank_ids_dem = []
    DEM_by_tank_tif_name = os.listdir(DEM_by_tank_output_path)
    for DEM_by_tank in DEM_by_tank_tif_name:
        tank_ids_dem.append([int(x) for x in regex.findall(DEM_by_tank)][0])

    #the tank ids with both lidar and DEMs
    tank_ids = list(set(tank_ids_dem).intersection(tank_ids_lidar))
    tank_ids = [str(i) for i in tank_ids]

    #paths to the DEM and lidar data 
    lidar_path_by_tank_for_height = []
    DEM_path_by_tank_for_height = []

    for tank_id in tank_ids:
        lidar_path_by_tank_for_height.append(os.path.join(lidar_by_tank_output_path, [string for string in lidar_by_tank_geojson_name if tank_id in string][0]))
        DEM_path_by_tank_for_height.append(os.path.join(DEM_by_tank_output_path, [string for string in DEM_by_tank_tif_name if tank_id in string][0]))
    return(tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height)

def add_bare_earth_data_to_lpc_by_tank_data(lidar_path_by_tank_for_height, DEM_path_by_tank_for_height):
    """
    Add average base elevation to dataframe
    """    
    for i, (lidar_path, DEM_path) in enumerate(zip(lidar_path_by_tank_for_height, DEM_path_by_tank_for_height)):
        # Read in each lidar dataset
        lidar = gpd.read_file(lidar_path)
        lidar_coords = [(x,y) for x, y in zip(lidar["X coordinate"], lidar["Y coordinate"])] #'EPSG:4326' coords

        # Open the DEM raster data and store metadata
        dem_src = rasterio.open(DEM_path)

        # Sample the raster at every point location and store values in DataFrame
        lidar['bare_earth_elevation'] = [z[0] for z in dem_src.sample(lidar_coords)]

        with open(lidar_path, "w") as file:
            file.write(lidar.to_json())
                   
def image_by_tank(tank_data, tiles_dir, output_path):
    image_path_list = []
    tank_data_grouped_by_tile = tank_data.groupby(tank_data.tile_name) #group gpds by dem

    for tile_name, tank_data_by_tile in tqdm.tqdm(tank_data_grouped_by_tile): 
        tile_path = os.path.join(tiles_dir, tile_name + ".tif")
        tile = cv2.imread(tile_path, cv2.IMREAD_UNCHANGED)

        for i, (tank_id, x_max, y_max, x_min, y_min) in enumerate(zip(tank_data_by_tile['id'], tank_data_by_tile['maxx_polygon_pixels'],
                                                                      tank_data_by_tile['maxy_polygon_pixels'], tank_data_by_tile['minx_polygon_pixels'],
                                                                      tank_data_by_tile['miny_polygon_pixels'])):
            # edit merge_tile_annotations function to ensure that annotation arrays are 2D
            if y_min == y_max:
                y_min -= 1
            if x_min == x_max:
                x_min -= 1
            
            tank_image = tile[y_min:y_max, x_min:x_max]                      
            tank_img_path = os.path.join(output_path, 'tank_image_tank_id_' + tank_id + '.tif') 
            image_path_list.append(tank_img_path)
            if not os.path.exists(tank_img_path):
                cv2.imwrite(tank_img_path, tank_image) #save images  
    return(image_path_list)
##########################################################################################################################
############################################   Height Estimation Figures    ##############################################
##########################################################################################################################
def add_titlebox(ax, text):
    ax.text(.55, .8, text,
    horizontalalignment='center',
    transform=ax.transAxes,
    bbox=dict(facecolor='white', alpha=0.6),
    fontsize=12.5)
    return ax

def height_estimation_figs(tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height, aerial_image_by_tank_for_height, plot_path):
    for i, (tank_id, lidar_path, DEM_path, aerial_image_path) in enumerate(zip(tank_ids, lidar_path_by_tank_for_height, DEM_path_by_tank_for_height,
                                                                               aerial_image_by_tank_for_height)):
        tank_id = str(tank_id)
        #Read in data 
        ##read in lidar
        lidar = gpd.read_file(lidar_path)
        #lidar = gpd.read_file(os.path.join(lidar_dir,"lidar_tank_id_"+tank_id+".geojson"))
        lidar["lpc_bee_difference"] = lidar["Z coordinate"]-lidar["bare_earth_elevation"]
        lidar.drop(lidar[(lidar['bare_earth_elevation'] ==-999999)].index, inplace=True) #remove no data values
        tank_class = lidar["object_class"].iloc[0]
        ##reproject for plotting
        wgs84 = pyproj.CRS('EPSG:4326')
        utm = pyproj.CRS(lidar["utm_projection"].iloc[0])
        Geometry = project_list_of_points(wgs84, utm, lidar["X coordinate"], lidar["Y coordinate"])
        x_y_utm = gpd.GeoDataFrame({'geometry': Geometry})
        X = gpd.GeoDataFrame(x_y_utm).bounds["minx"]
        Y = gpd.GeoDataFrame(x_y_utm).bounds["miny"]
        ##read in dem
        dem_test = rasterio.open(DEM_path) 
        dem = dem_test.read(1)
        dem[dem==-999999] = np.nan
        dem_test.close()
        ##Read in aerial imagery
        img = cv2.imread(aerial_image_path)

        #Make figure
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Estimating Height Using Elevation Data for Object #'+tank_id+" of class "+tank_class, fontsize=20)
        #Define ranges and colors for colorbar, and for the image
        ##min and max values
        vmin=lidar[["Z coordinate","bare_earth_elevation","lpc_bee_difference"]].min().min()
        vmax=lidar[["Z coordinate","bare_earth_elevation","lpc_bee_difference"]].max().max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ##get colorbar
        current_cmap = mpl.cm.get_cmap('terrain').copy()
        current_cmap.set_bad(color='red')
        #make grided plot
        widths = [.5, .5, .5, .5, .5, .5, .5, .5, 2]
        heights = [.5, .5, .5, .5, .5, .5, .5, .5, 0.1]
        gs = gridspec.GridSpec(nrows=len(heights), ncols=len(heights), width_ratios=widths,height_ratios=heights)
        #plot raw data
        ## Image
        ax_img = plt.subplot(gs[:2, 1:3])
        ax_img.imshow(img, cmap=current_cmap, norm=norm, aspect="auto")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title('NAIP Imagery', fontsize=10)
        #ax_img.set_aspect('auto', adjustable='box')  # NEW
        ##DEM
        ax_dem = plt.subplot(gs[:2, 3:5])
        ax_dem.imshow(dem, cmap=current_cmap, norm=norm, aspect="auto")
        ax_dem.set_xticks([])
        ax_dem.set_yticks([])
        ax_dem.set_title('Digital Elevation Model', fontsize=10)
        #ax_dem.set_aspect(asp)
        #BEE
        ax_bee = plt.subplot(gs[2:4, 1:3])
        ax_bee.scatter(X,Y, c=lidar["bare_earth_elevation"], cmap=current_cmap, norm=norm)
        ax_bee.set_xticks([])
        ax_bee.set_yticks([])
        ax_bee.set_title('Bare Earth Elevation', fontsize=10)
        asp = np.diff(ax_bee.get_xlim())[0] / np.diff(ax_bee.get_ylim())[0]
        #ax_bee.set_aspect(asp)
        ##LPC
        ax_lpc = plt.subplot(gs[2:4, 3:5])
        ax_lpc.scatter(X,Y, c=lidar["Z coordinate"], cmap=current_cmap, norm=norm)
        ax_lpc.set_xticks([])
        ax_lpc.set_yticks([])
        ax_lpc.set_title('Lidar Point Cloud', fontsize=10)
        ax_lpc.set_xlim(ax_bee.get_xlim())
        ax_lpc.set_ylim(ax_bee.get_ylim())

        #ax_lpc.set_aspect(asp)
        #Difference between DSM and DEM over all values bounding box
        H = round(lidar["lpc_bee_difference"].mean(),2)
        ax_bboxdist = plt.subplot(gs[4:6, :2])
        ax_bboxdist.set_title('LPC - DEM', fontsize=10)
        add_titlebox(ax_bboxdist, '(H ='+str(H)+'m)')
        ax_bboxdist.scatter(X,Y, c=lidar["lpc_bee_difference"], cmap=current_cmap, norm=norm)
        ax_bboxdist.set_xticks([])
        ax_bboxdist.set_yticks([])
        #axbboxdist.set_aspect(asp)
        #ax_bboxdist.set_aspect('auto', adjustable='box')
        ax_bboxdist.set_xlim(ax_bee.get_xlim())
        ax_bboxdist.set_ylim(ax_bee.get_ylim())
        #Difference between DSM and DEM for all DSM values greater than the 25th quantile 
        Q25 = lidar["Z coordinate"].quantile(.25) 
        idxs = np.where(lidar["Z coordinate"] > Q25)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_Q25 = plt.subplot(gs[4:6, 2:4])
        ax_Q25.set_title('LPC (over Q25['+str(round(Q25,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_Q25, '(H ='+str(H)+'m)')
        ax_Q25.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_Q25.set_xticks([])
        ax_Q25.set_yticks([])
        ax_Q25.set_xlim(ax_bee.get_xlim())
        ax_Q25.set_ylim(ax_bee.get_ylim())
        #axQ25.set_aspect(asp)
        #Difference between DSM and DEM for all DSM values greater than the mean
        mean = lidar["Z coordinate"].mean()
        idxs = np.where(lidar["Z coordinate"] > mean)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_mean = plt.subplot(gs[4:6, 4:6])
        ax_mean.set_title('LPC (over mean ['+str(round(mean,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_mean, '(H ='+str(H)+'m)')
        ax_mean.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])
        ax_mean.set_xlim(ax_bee.get_xlim())
        ax_mean.set_ylim(ax_bee.get_ylim())
        #axmean.set_aspect(asp)             
        #Difference between DSM and DEM for all DSM values greater than the median
        median = lidar["Z coordinate"].median()
        idxs = np.where(lidar["Z coordinate"] > median)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_median = plt.subplot(gs[4:6, 6:8])
        ax_median.set_title('LPC (over median ['+str(round(median,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_median, '(H ='+str(H)+'m)')
        ax_median.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_median.set_xticks([])
        ax_median.set_yticks([])
        ax_median.set_xlim(ax_bee.get_xlim())
        ax_median.set_ylim(ax_bee.get_ylim())
        #axmedian.set_aspect(asp)
        #Difference between DSM and DEM for all DSM values greater than the 75th quantile 
        Q75 = lidar["Z coordinate"].quantile(.75) 
        idxs = np.where(lidar["Z coordinate"] > Q75)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_Q75 = plt.subplot(gs[6:8, :2])
        ax_Q75.set_title('LPC (over Q75 ['+str(round(Q75,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_Q75, '(H ='+str(H)+'m)')
        ax_Q75.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_Q75.set_xticks([])
        ax_Q75.set_yticks([])
        ax_Q75.set_xlim(ax_bee.get_xlim())
        ax_Q75.set_ylim(ax_bee.get_ylim())
        #axQ75.set_aspect(asp)
        #Difference between DSM and DEM for all DSM values greater than the 90th quantile 
        Q90 = lidar["Z coordinate"].quantile(.90) 
        idxs = np.where(lidar["Z coordinate"] > Q90)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_Q90 = plt.subplot(gs[6:8, 2:4])
        ax_Q90.set_title('LPC (over Q90 ['+str(round(Q90,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_Q90, '(H ='+str(H)+'m)')
        ax_Q90.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_Q90.set_xticks([])
        ax_Q90.set_yticks([])
        #ax_Q90.set_aspect(asp)
        ax_Q90.set_xlim(ax_bee.get_xlim())
        ax_Q90.set_ylim(ax_bee.get_ylim())
        #Difference between DSM and DEM for all DSM values greater than the 95th quantile 
        Q95 = lidar["Z coordinate"].quantile(.95) 
        idxs = np.where(lidar["Z coordinate"] > Q95)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_Q95 = plt.subplot(gs[6:8, 4:6])
        ax_Q95.set_title('LPC (over Q95 ['+str(round(Q95,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_Q95, '(H ='+str(H)+'m)')
        ax_Q95.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_Q95.set_xticks([])
        ax_Q95.set_yticks([])
        ax_Q95.set_xlim(ax_bee.get_xlim())
        ax_Q95.set_ylim(ax_bee.get_ylim())
        #Difference between DSM and DEM for all DSM values greater than the 99th quantile 
        Q99 = lidar["Z coordinate"].quantile(.99) 
        idxs = np.where(lidar["Z coordinate"] > Q99)[0]
        H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
        ax_Q99 = plt.subplot(gs[6:8, 6:8])
        ax_Q99.set_title('LPC (over Q99 ['+str(round(Q99,2))+']) - DEM', fontsize=10)
        add_titlebox(ax_Q99, '(H ='+str(H)+'m)')
        ax_Q99.scatter(X.iloc[idxs], Y.iloc[idxs], c=lidar["lpc_bee_difference"].iloc[idxs], cmap=current_cmap, norm=norm)
        ax_Q99.set_xticks([])
        ax_Q99.set_yticks([])
        ax_Q99.set_xlim(ax_bee.get_xlim())
        ax_Q99.set_ylim(ax_bee.get_ylim())
        #Distribution of LPC
        ax_hist = plt.subplot(gs[1:7, 8])
        #axhist.set_aspect('equal', adjustable='box')  # NEW
        ax_hist.set_title('LPC Distribution', fontsize=10)
        ax_hist.axvline(x = lidar["Z coordinate"].quantile(1/4), color = 'orange', label = '25% Q')
        ax_hist.axvline(x = median, color = 'red', label = 'median')
        ax_hist.axvline(x = mean, color = 'black', label = 'mean')
        ax_hist.axvline(x = lidar["Z coordinate"].mode()[0], color = 'purple', label = 'mode')
        ax_hist.axvline(x = Q75, color = 'orange', label = '75% Q')
        ax_hist.hist(lidar["Z coordinate"], bins = int(len(lidar["Z coordinate"])/100),
                    color = 'blue', edgecolor = 'blue')
        ax_hist.set_yticks([])
        #axhist.set_aspect('auto', adjustable='box')  # NEW
        ax_hist.legend(loc="upper left")
        #Add in color bar
        cbax = plt.subplot(gs[8, 0:8])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=current_cmap),
                          cax=cbax, orientation='horizontal' )#use the defined variables cmap and norm
        cb.ax.tick_params(labelsize=10) #set ticks
        cb.set_label('Elevation (m)',fontsize=12) #label colorbar
        plt.tight_layout()
        #plot
        path = os.path.join(plot_path, tank_class)
        os.makedirs(path, exist_ok = True)
        fig.savefig(os.path.join(path, tank_id+".jpg"))
        plt.close(fig)

def add_height_estimate_by_tank_data(tank_ids, lidar_path_by_tank_for_height, tank_data):
    """
    Add average base elevation to dataframe
    """    
    heights = [-999999] * len(tank_data)
    for i, (tank_id, lidar_path) in enumerate(zip(tank_ids, lidar_path_by_tank_for_height)):
        tank_id = int(tank_id)
        #Read in data 
        ##read in lidar data for object
        lidar = gpd.read_file(lidar_path)
        #get tank class
        tank_class = lidar["object_class"].iloc[0]
        #calculate difference between LPC and DEM(Bare earth elevation)
        lidar["lpc_bee_difference"] = lidar["Z coordinate"] - lidar["bare_earth_elevation"]
        #remove no data values
        lidar.drop(lidar[(lidar['bare_earth_elevation'] ==-999999)].index, inplace=True) #remove no data values
        if tank_class == "closed_roof_tank":
            Q75 = lidar["Z coordinate"].quantile(.75) 
            idxs = np.where(lidar["Z coordinate"] > Q75)[0]
            H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
            heights[tank_id] = H
        if tank_class == "external_floating_roof":
            Q95 = lidar["Z coordinate"].quantile(.95) 
            idxs = np.where(lidar["Z coordinate"] > Q95)[0]
            H = round(lidar["lpc_bee_difference"].iloc[idxs].mean(),2)
            heights[tank_id] = H
    tank_data["heights"] = heights
    return(tank_data)

def write_gdf(gdf, output_filepath, output_filename = 'tile_level_annotations'):
    """
    write tank data 
    """
    gdf.crs = "EPSG:4326" #assign projection

    #save geodatabase as json
    with open(os.path.join(output_filepath, output_filename+".json"), 'w') as file:
        file.write(gdf.to_json()) 

    ##save geodatabase as geojson 
    with open(os.path.join(output_filepath, output_filename+".geojson"), "w") as file:
        file.write(gdf.to_json()) 

    ##save geodatabase as shapefile
    gdf_shapefile = gdf.drop(columns=["chip_name","polygon_vertices_pixels","polygon_vertices_lon_lat"])
    gdf_shapefile.to_file(os.path.join(output_filepath,output_filename+".shp"))
############################################################################################
####################    Shadow Detection height estimation   ###############################
############################################################################################
def calculate_diameter(bbox, resolution = 0.6):
    """ Calculate the diameter of a given bounding bbox for imagery of a given resolution
    Arg:
    bbox(list): a list of the (xmin, ymin, xmax, ymax) coordinates for box 
    resolution(float): the (gsd) resolution of the imagery
    Returns:
    (diameter): the diameter of the bbox of interest
    """
    obj_xmin, obj_ymin, obj_xmax, obj_ymax = bbox
    obj_width = obj_xmax - obj_xmin
    obj_height = obj_ymax - obj_ymin
    diameter = min(obj_width, obj_height) * resolution #meter
    return(diameter)

def tile_annotation_to_dictionary(tile_name, tile_annotation, label_map):
    "create a dictionary of the bbox and labels for the tile level annotations geopandas dataframe"
    bboxes = []
    labels = []
    for i in tile_annotation.index:
        x_min = tile_annotation['minx_polygon_pixels'][i]
        y_min = tile_annotation['miny_polygon_pixels'][i]
        x_max = tile_annotation['maxx_polygon_pixels'][i]
        y_max = tile_annotation['maxy_polygon_pixels'][i]
        bboxes.append([x_min,y_min,x_max,y_max])
        labels.append(label_map[tile_annotation["object_class"][i]])
    return {'bboxes': bboxes, 'labels': labels}

def convert(o):
    """Issue raised when dumping numpy.int64 into json string in Python 3.
    There is a workaround provided by Serhiy Storchaka
    Reference links:
    https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    https://bugs.python.org/issue24313
    """
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def tile_level_annotations_gpds_to_json(path, label_map):
    tile_paths = []
    tile_annotations = []
    #read in tile level annotation pandas 
    tile_level_annotations = gpd.read_file(os.path.join(path,'tile_level_annotations/tile_level_annotations.geojson'))
    tile_level_annotations = tile_level_annotations[['tile_name',"object_class",'minx_polygon_pixels','miny_polygon_pixels', 'maxx_polygon_pixels','maxy_polygon_pixels']]
    tile_names = np.unique(tile_level_annotations['tile_name']) #get unique tile names
    #get path and annotations by tile 
    for tile_name in tile_names:
        tile_annotation = tile_level_annotations[tile_level_annotations["tile_name"] == tile_name]
        tile_annotations.append(tile_annotation_to_dictionary(tile_name, tile_level_annotations, label_map))
        tile_paths.append(os.path.join(path, "tiles", tile_name + ".tif"))
    #write annotations and paths to json
    with open(os.path.join(path, 'tile_annotation.json'), 'w') as outfile:
        json.dump(tile_annotations, outfile, default=convert)
    with open(os.path.join(path, 'tile_images.json'), 'w') as outfile:
        json.dump(tile_paths, outfile, default=convert)
        
def check_bb(bbox, h, w, c):
    """
    The algorithm is designed to work with tanks that are fully in frame. Bounding boxes that reach the edge of an image (indicating the tank extends beyond the image) are excluded from processing.
    Check the distance between the bounding box and the edge of image. If the bounding box is within two pixels, return False, otherwise return True.
    Args:
    bbox(list): a boxx in [x_min,y_min,x_max,y_max] format
    h, w, c(int): the height, width, and depth of the image containing the bounding boxes of interest
    """
    x_min,y_min,x_max,y_max = bbox
    for d in bbox:
        if x_min <= 2 or x_max >= w-2:
            return False
        elif y_min <=2 or y_max >= h-2:
            return False
    return True
def intersection(bb1, bb2):
    """
    intersection` calculates the pixel area intersection between two bounding boxes
    """
    x_min1, y_min1, x_max1, y_max1  = bb1
    x_min2, y_min2, x_max2, y_max2 = bb2
    
    x_left = max(x_min1, x_min2)
    x_right = min(x_max1, x_max2)
    y_top = max(y_min1, y_min2)
    y_bottom = min(y_max1, y_max2)

    intersection = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top+1)
    return intersection