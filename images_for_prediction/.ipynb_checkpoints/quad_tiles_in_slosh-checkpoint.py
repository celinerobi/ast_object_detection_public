import geopandas as gpd
import fiona
import os
import geopandas as gpd
import pandas as pd
from glob import glob
import shutil


def unzip_read_shp(map_indexes_path, quad_dir):
    os.makedirs(quad_dir, exist_ok=True)
    shutil.unpack_archive(map_indexes_path, quad_dir)
    data_path = glob(quad_dir + "/*.shp")[0]
    return gpd.read_file(data_path)


def combine_slosh_extents(slosh_extent_dir):
    #read in and combine slosh extent
    combined_extent = []
    for file_name in os.listdir(slosh_extent_dir):
        os.path.join(file_name, slosh_extent_dir)
    slosh_extent_file_paths = glob(slosh_extent_dir + "/*.shp")
    slosh_extent_gdf = pd.concat([gpd.read_file(p) for p in slosh_extent_file_paths], 
                                ignore_index=True)
    return slosh_extent_gdf

    
def get_args_parse():
    parser = argparse.ArgumentParser(
        description='Identify quad indexs within slosh modeled area')
    parser.add_argument('--quad_dir', type=str, default="/work/csr33/quads/map_indexes_QD12K",
                        help='quad directory')
    parser.add_argument('--map_indexes_path', type=str, 
                        default="/work/csr33/quads/map_indexes_QD12K_extract_4130469_01.zip",
                        help='path to quad indexes')
    parser.add_argument('--slosh_extent_dir', type=str, help='Directory holding slosh extent datasets',
                       default="/hpc/group/borsuklab/nat_hazard_data/slosh/unzip/v3_extent_shp")
    parser.add_argument('--subset_quad_path', type=str, 
                        default="/work/csr33/quads/subset_quad_index_to_slosh.geojson",
                        help='Filename to save tile names and urls')
    args = parser.parse_args()
    return args


def main(args):
    #read in quad data
    quad_gpd = unzip_read_shp(args.map_indexes_path, args.quad_dir)
    #read in combine slosh extent gdf
    slosh_extent_gdf = combine_slosh_extents(args.slosh_extent_dir)
    #reproject slosh extent to map_indexes
    slosh_extent_gdf = slosh_extent_gdf.to_crs(quad_gpd.crs)
    #identify quads in slosh extent
    subset_quad_index_to_slosh = gpd.sjoin(quad_gpd, slosh_extent_gdf, 
                                           how='inner', predicate='intersects')
    #write to file
    subset_quad_index_to_slosh.to_file(args.subset_quad_path, driver='GeoJSON')
    

if __name__ == '__main__':
    ### Get the arguments 
    args = get_args_parse()
    main(args)