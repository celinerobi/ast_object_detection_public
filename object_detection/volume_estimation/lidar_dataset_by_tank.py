help('modules')

import tqdm
import numpy as np
import pandas as pd
import os
from glob import glob

import geopandas as gpd #important
import volume_estimation_functions as vol_est
import argparse

def get_args_parse():
    parser = argparse.ArgumentParser(description='This script adds LPC data to tile level tank data')
    parser.add_argument('--tank_and_lidar_data_dir', type=str, default=None,
                        help='path to folder containing tank_and_lidar_dir')
    parser.add_argument('--lidar_by_tank_output_path', type=str, default=None, 
                        help='path to hold, lidar by tank geojson data')

    args = parser.parse_args()
    return args

def main(args):
    tank_and_lidar_data_paths = glob(args.tank_and_lidar_data_dir + "/*.geojson")
    #seperate lidar data by tank
    for tank_and_lidar_data in tqdm.tqdm(tank_and_lidar_data_paths):
        print(tank_and_lidar_data)
        tank_and_lidar_data = gpd.read_file(tank_and_lidar_data)

        if len(tank_and_lidar_data["geometry"]) == 0: #skip over empty dataframes
            print("empty dataframe")
            continue

        print("group data")

        tank_and_lidar_data_grouped = tank_and_lidar_data.groupby(tank_and_lidar_data.id)
        for id_, lidar_by_tank in tank_and_lidar_data_grouped:
            #get output filename
            output_filename = os.path.join(args.lidar_by_tank_output_path, "lidar_tank_id_" + id_ + ".geojson")
            ###Note: merge multiple lidar datasets ###
            if os.path.exists(output_filename):
                existing_lidar_by_tank = gpd.read_file(output_filename)
                lidar_by_tank = pd.concat([lidar_by_tank, existing_lidar_by_tank])
                lidar_by_tank = lidar_by_tank.drop_duplicates()

            #write to geojson
            with open(output_filename, "w") as file:
                file.write(lidar_by_tank.to_json()) 
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)
