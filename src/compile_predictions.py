import os
import argparse
from glob import glob
import pandas as pd
import numpy as np
import geopandas as gpd    
from copy import copy
# https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook
# https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
        
def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Identify quad indexs within slosh modeled area')
    
    parser.add_argument("--detect_tank_dir", type=str)
    parser.add_argument("--compiled_data_path", type=str)
    args = parser.parse_args()
    return args

        
def compile_data(args):
    #compile data
    detected_tank_paths = glob(args.detect_tank_dir + "/*")
    tables = [gpd.read_parquet(f) for f in detected_tank_paths]
    detected_tanks_gdf = gpd.GeoDataFrame(pd.concat(tables, ignore_index=True), crs=tables[0].crs)
    print(len(detected_tanks_gdf))
    # add index column
    detected_tanks_gdf["id"] = detected_tanks_gdf.index
    #estimate tank capacity for external and closed roof tanks
    detected_tanks_gdf["capacity"] = np.nan 
    
    detected_tanks_gdf.loc[detected_tanks_gdf['height'] >= 50, 'height'] = 50

    #estimate capacity
    can_estimate_capacity = (~detected_tanks_gdf['height'].isnull()) &\
                            (detected_tanks_gdf['class_name'] != "spherical") &\
                            (detected_tanks_gdf['height'] > 1.0)

    has_height_gdf = copy(detected_tanks_gdf[can_estimate_capacity])
    has_height_gdf["capacity"] = np.pi * (has_height_gdf["diameter"] / 2) ** 2 * has_height_gdf["height"]
    detected_tanks_gdf.loc[can_estimate_capacity, 'capacity'] = has_height_gdf['capacity']
    #save in formats for add slosh
    detected_tanks_gdf.rename(columns = {'class_name':'object_class'}, inplace = True) 
    #detected_tanks_gdf.to_parquet("/hpc/group/borsuklab/csr33/object_detection/compiled_predicted_tank.parquet")
    detected_tanks_gdf.to_file(args.compiled_data_path, driver='GeoJSON') 
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    compile_data(args)
    
    