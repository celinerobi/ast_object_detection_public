import os
import argparse

import pandas as pd
import geopandas as gpd    

import pystac_client
import planetary_computer

# https://planetarycomputer.microsoft.com/dataset/naip#Example-Notebook
# https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/
def split_catalog_id(idx):
    state, full_name = idx.split("_",1)
    quad_id = '_'.join(full_name.split('_')[:3])
    return pd.Series((full_name, quad_id))
    

def unzip_read_shp(map_indexes_path, quad_dir):
    os.makedirs(quad_dir, exist_ok=True)
    shutil.unpack_archive(map_indexes_path, quad_dir)
    data_path = glob(quad_dir + "/*.shp")[0]
    return gpd.read_file(data_path)

    

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
    parser.add_argument('--naip_data_path', type=str, 
                        default="/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet",
                        help='Path to store NAIP information path')
    parser.add_argument("--chunked_naip_data_dir", default="/work/csr33/images_for_predictions/chunked_naip_data", type=str)
    parser.add_argument("--chunked_naip_data_filename", default="chunked_naip_data", type=str)
    parser.add_argument("--num_chunks", default=100, type=int)

    args = parser.parse_args()
    return args

        
def extract_naip_data(args):
    
    # bulk request data from planetary computing
    #https://planetarycomputer.microsoft.com/docs/quickstarts/stac-geoparquet/
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/",
                                        modifier=planetary_computer.sign_inplace,)
    asset = catalog.get_collection("naip").assets["geoparquet-items"]
    naip_df = gpd.read_parquet(asset.href, storage_options=asset.extra_fields["table:storage_options"])
    # add in formated image name and quad id
    naip_df[["img_name","quad_id"]] = naip_df["id"].apply(split_catalog_id)
    # Sort values by 'date' in descending order so the newest date is first
    naip_df = naip_df.sort_values('naip:year', ascending=False)
    # Drop duplicates on column 'x', keeping the first occurrence (which is the newest due to sorting)
    naip_df = naip_df.drop_duplicates(subset='quad_id', keep='first')

    
    # read in slosh extent data
    slosh_extent_gdf = gpd.read_file(args.slosh_extent_path).to_crs(naip_df.crs)
    #subset slosh extent to include only areas in CONUS
    country_boundary = gpd.read_file(args.country_boundary_path).to_crs(naip_df.crs)
    slosh_extent_in_conus = gpd.clip(slosh_extent_gdf, country_boundary)    
    
    
    #subset naip data by slosh data
    subset_naip_in_slosh = gpd.sjoin(naip_df, slosh_extent_in_conus, how='inner', predicate='intersects')
    subset_naip_in_slosh["img_url"] = subset_naip_in_slosh.assets.apply(lambda asset: asset["image"]["href"])#
    subset_naip_in_slosh.to_parquet(args.naip_data_path)
    return subset_naip_in_slosh


if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    subset_naip_in_slosh = extract_naip_data(args)
    os.makedirs(args.chunked_naip_data_dir, exist_ok=True)
    chunk_df(subset_naip_in_slosh, args)
    
    