import pystac_client
import planetary_computer
import geopandas as gpd
import stackstac
import rasterio
from rasterio.merge import merge
import glob
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from rasterio import mask
from rasterio import plot
import shapely
import time


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
        mask = clipped_image == -9999
        masked_arr = np.ma.masked_where(mask, clipped_image).squeeze()
    
        h.append(np.quantile(masked_arr.data.flatten(), 0.9))
        w.append(clipped_image.size)
    [raster.close() for raster in rasters] #close rasters
    # average height
    if len(h) > 0:
        return np.average(h, weights=w)
    else:
        return None


def height_estimation(ast_data, collection):
    start = time.time()
    height_list = [] #height list
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier = planetary_computer.sign_inplace,)
    
    for tank_id, row in ast_data.iterrows():
        #create utm geometry
        minx, miny, maxx, maxy = row[['nw_x_utm_object_coord','se_y_utm_object_coord',
                                      'se_x_utm_object_coord','nw_y_utm_object_coord']]
        tank_geometry = shapely.geometry.box(minx, miny, maxx, maxy, ccw=True) #utm
        #search catalog using lat lon geometry
        item_collection = catalog.search(collections=[collection], 
                                intersects=row.geometry.buffer(0.001)).item_collection()

        if len(item_collection) > 0:
            rasters = read_raster(item_collection)
            # ensure tank the data intersects
            rasters = ensure_raster_tank_intersect(rasters, tank_geometry)
            # calculate height
            height = calculate_height(rasters, tank_geometry)
            height_list.append(height)
        else:
            height_list.append(None) 
   
    print(time.time() - start)
    return height_list


def get_args_parse():
    parser = argparse.ArgumentParser("Height Estimation")
    parser.add_argument("--ast_data_path", type=str, help="path to training weights for trained model")
    parser.add_argument("--collection", type=str, help="the name of the planetary computer collection")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    ast_data = gpd.read_file(args.ast_data_path)
    asta_data["height"] = height_estimation(ast_data, args.collection)