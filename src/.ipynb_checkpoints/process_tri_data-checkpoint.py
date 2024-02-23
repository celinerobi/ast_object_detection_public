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
    s = ''.join([c for c in s[:12] if c.isdigit() or c in '.' or c in "-"])
    s = s.split("-")

    s = [float(s) for s in s]
    s = np.average(s)
    return s
    
def convert_chemical_data_to_sg(tri_2022_us_chemical, chemical_data):
    sg_data = []
    for cas_number in tri_2022_us_chemical["37. CAS#"]:
        if cas_number is None:
            sg_data.append(None)
        elif cas_number is np.NaN:
            sg_data.append(None)
        else:
            subset_of_chemical_data_by_tank = chemical_data[chemical_data['CAS#'].isin(cas_number)]
            if len(subset_of_chemical_data_by_tank) > 0:
                subset_of_chemical_data_by_tank = subset_of_chemical_data_by_tank["Specific gravity"].to_list()
                sg_data.append(subset_of_chemical_data_by_tank)
            else:
                sg_data.append(None)
    return sg_data
    

def chunk_df(naip_df, args):
    # Calculate the number of rows per chunk
    rows_per_chunk = len(naip_df) // args.num_chunks
    df_chunks = [naip_df.iloc[i : i + rows_per_chunk] for i in range(0, len(naip_df), rows_per_chunk)]
    for i, chunk in enumerate(df_chunks):
        file_path = f'{args.chunked_naip_data_dir}/{args.chunked_naip_data_filename}_{i}.parquet'
        chunk.to_parquet(file_path, index=False)
        
        
def get_args_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--tri_2022_us_path", default="/hpc/group/borsuklab/csr33/chemical_data/tri/2022_us.csv", type=str)
    parser.add_argument("--naics_industry_codes_path", default="/hpc/home/csr33/spatial-match-ast-chemicals/naics_industry_keys.csv", type=str)
    parser.add_argument('--tri_with_sg_path', type=str, 
                        default="/hpc/home/csr33/tri_with_specific_gravity.parquet")
    args = parser.parse_args()
    return args

        
def process_chemical_data(args):
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
    
    # read in and format chemical data
    chemical_data = pd.read_csv("/hpc/group/borsuklab/csr33/chemical_data/niosh_pocket_guide/NIOSH Pocket Guide.csv")
    # remove rows where specific gravity is none
    chemical_data = chemical_data.dropna(subset=['Specific gravity'])
    # remove rows where specific gravity is for a metal
    # Remove rows with 'metal'
    chemical_data = chemical_data[~chemical_data['Specific gravity'].str.contains('metal', case=False)]
    chemical_data = chemical_data[~chemical_data['Specific gravity'].isin(["?", "? "])]
    chemical_data['Specific gravity'] = [average_str(sg) for sg in chemical_data['Specific gravity']]
    
    #add sg to chemical data
    tri_2022_us_chemical["specific_gravity"] = convert_chemical_data_to_sg(tri_2022_us_chemical, chemical_data)
    tri_2022_us_chemical = gpd.GeoDataFrame(tri_2022_us_chemical)
    tri_2022_us_chemical['34. CHEMICAL'] = tri_2022_us_chemical['34. CHEMICAL'].apply(lambda x: str(x))
    tri_2022_us_chemical['37. CAS#'] = tri_2022_us_chemical['37. CAS#'].apply(lambda x: str(x))
    tri_2022_us_chemical['specific_gravity'] = tri_2022_us_chemical['specific_gravity'].apply(lambda x: str(x))

    #write to file
    #tri_2022_us_chemical.to_file(args.tri_with_sg_path, driver="GeoJSON")  
    tri_2022_us_chemical.to_parquet(args.tri_with_sg_path)  

    
if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    process_chemical_data(args)
    
    