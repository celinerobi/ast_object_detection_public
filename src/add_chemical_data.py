import os
import argparse

import pandas as pd
import numpy as np
import geopandas as gpd    

from shapely.geometry import Point
from sklearn.neighbors import BallTree

# https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook
# https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
def average_str(s):
    s = s.split(" (")[0].split("-")
    s = [float(s) for s in s]
    s = np.average(s)
    
def add_sg_data_to_tank_data(tank_data, chemical_data):
    sg_data = []
    for cas_number in tank_data["cas_number"]:
        if cas_number is None:
            sg_data.append(None)
        elif cas_number is np.NaN:
            sg_data.append(None)
        else:
            subset_of_chemical_data_by_tank = chemical_data[chemical_data['CAS#'].isin(cas_number)]
            if len(subset_of_chemical_data_by_tank) > 0:
                sg_data.append(subset_of_chemical_data_by_tank)
            else:
                sg_data.append(None)
    tank_data["sg"] = sg
    return tank_data
    

def chunk_df(naip_df, args):
    # Calculate the number of rows per chunk
    rows_per_chunk = len(naip_df) // args.num_chunks
    df_chunks = [naip_df.iloc[i : i + rows_per_chunk] for i in range(0, len(naip_df), rows_per_chunk)]
    for i, chunk in enumerate(df_chunks):
        file_path = f'{args.chunked_naip_data_dir}/{args.chunked_naip_data_filename}_{i}.parquet'
        chunk.to_parquet(file_path, index=False)
        
        
def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Identify quad indexs within slosh modeled area')
    parser.add_argument("--height_estimation_dir", type=str, help="path to the directory storing height estimation on predictions")
    parser.add_argument("--complete_predicted_data_dir", type=str, help="path to the directory storing predictions with all data")
    parser.add_argument("--tri_with_sg_path", type=str)
    parser.add_argument("--chunk_id",  type=int)
    args = parser.parse_args()
    return args

        
def sg(args):
    # read in data for detected tanks
    detected_tanks = gpd.read_parquet(os.path.join(args.height_estimation_dir, 
                                f"merged_predictions_height_{args.chunk_id}.parquet"))

    # Read in tri data #/work/csr33/spatial_matching/tri/
    tri_sg = gpd.read_parquet(args.tri_with_sg_path)

    # Create a BallTree for quick nearest neighbor lookup
    btree = BallTree(tri_sg.geometry.apply(lambda x: (x.x, x.y)).tolist()) 

    # Find closest point for each polygon
    closest_points = []
    for polygon in detected_tanks.geometry:
        point = polygon.representative_point()
        # Query ball tree to find closest point
        dist, idx = btree.query([(point.x, point.y)], k=1) 
        if dist > 0.1: 
            # No point within 1 km
            closest_points.append(None)
        else:
            closest_points.append(tri_sg.geometry.iloc[idx[0][0]])

    # Add closest points back to polygons GeoDataFrame
    detected_tanks['closest_point'] = closest_points

    #add chemical data to tile level annotations
    merged_df = pd.merge(detected_tanks, tri_sg, left_on='closest_point', right_on='geometry', how='left')
    detected_tanks[["chemical_name", "cas_number", "facility_name"]] = merged_df[["34. CHEMICAL", "37. CAS#", "4. FACILITY NAME"]]
    print(type(detected_tanks))
    detected_tanks.drop(columns=["closest_point"], inplace=True)
    detected_tanks.to_parquet(os.path.join(args.complete_predicted_data_dir, f"complete_predictions_{args.chunk_id}.parquet"))
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    sg(args)
    
    