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
    parser.add_argument('--slosh_extent_path', type=str, help='Path to largest slosh extent datasets',
                       default="/hpc/group/borsuklab/nat_hazard_data/slosh/unzip/v3_extent_shp/cat1_merge_Dissolve.shp")
    parser.add_argument('--country_boundary_path', type=str, 
                        default="/hpc/group/borsuklab/political-boundaries/states/generalized-conus-country-boundary.shp",
                        help='Tracts in slosh')

    
    parser.add_argument("--detected_tanks_path", type=str)
    parser.add_argument("--naics_industry_codes_path", default="/hpc/home/csr33/spatial-match-ast-chemicals/naics_industry_keys.csv", type=str)
    parser.add_argument("--tri_2022_us_path", default="/hpc/group/borsuklab/csr33/chemical_data/tri/2022_us.csv", type=str)
    parser.add_argument('--detected_tanks_with_chemical_data_path', type=str, help='')
    args = parser.parse_args()
    return args

        
def dataset_for_object_detection(args):
    # Read in tri data #/work/csr33/spatial_matching/tri/
    tri_2022_us = pd.read_csv(args.tri_2022_us_path)
    #add geometries
    Geometry = [Point(xy) for xy in zip(tri_2022_us['13. LONGITUDE'], tri_2022_us['12. LATITUDE'])] 
    tri_2022_us = gpd.GeoDataFrame(tri_2022_us, crs="EPSG:4326", geometry=Geometry)

    #remove rows without locations
    tri_2022_us = tri_2022_us[~np.isnan(tri_2022_us["12. LATITUDE"])]
    tri_2022_us = tri_2022_us[~np.isnan(tri_2022_us["13. LONGITUDE"])]

    #subset tri data based on naics codes
    naics_industry_codes = pd.read_csv(args.naics_industry_codes_path)
    tri_2022_us =tri_2022_us[tri_2022_us["19. INDUSTRY SECTOR CODE"].isin(naics_industry_codes["2022 NAICS US Code"].tolist())]

    # Group by 'geometry' and aggregate 'name' and 'value' into lists
    tri_2022_us_chemical = tri_2022_us.groupby('geometry').agg({"34. CHEMICAL": list, "37. CAS#": list}).reset_index()

    # Get unique tri locations
    unique_tri_2022_us_values, unique_tri_2022_us_indices =  np.unique(tri_2022_us[["12. LATITUDE","13. LONGITUDE"]].values, return_index= True, axis = 0)
    tri_2022_us_unique_locations = tri_2022_us.iloc[unique_tri_2022_us_indices]
    
    # Create a BallTree for quick nearest neighbor lookup
    btree = BallTree(tri_2022_us_unique_locations.geometry.apply(lambda x: (x.x, x.y)).tolist()) 

    # read in data for detected tanks
    detected_tanks = gpd.read_file(args.detected_tanks_path)
    
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
            closest_points.append(tri_2022_us_unique_locations.geometry.iloc[idx[0][0]])

    # Add closest points back to polygons GeoDataFrame
    detected_tanks['closest_point'] = closest_points

    #add chemical data to tile level annotations
    merged_df = pd.merge(detected_tanks, tri_2022_us_chemical, left_on='closest_point', right_on='geometry', how='left')
    detected_tanks[["chemical_name", "cas_number"]] = merged_df[["34. CHEMICAL", "37. CAS#"]]
    detected_tanks.to_csv(args.detected_tanks_with_chemical_data_path)
    
    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    dataset_for_object_detection(args)
    
    