import argparse
import math

import pandas as pd
import numpy as np

import cv2

import rioxarray

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster




def data_processing(url, item_dim=640):
    #read in image
    tile = rioxarray.open_rasterio(url)#.squeeze()
    tile_channels, tile_height, tile_width = tile.shape  # the size of the tile
    # divide the tile into item_dim by item_dim chips (rounding up)
    row_index = math.ceil(tile_height / item_dim)
    col_index = math.ceil(tile_width / item_dim)   
    # determine slices for tile
    slice_indices = [[[x*item_dim, x*item_dim+item_dim], [y*item_dim, y*item_dim+item_dim]] for y in range(row_index) for x in range(col_index)]
    #split tile data array
    split_tile = [tile.isel(x=slice(s[0][0], s[0][1]), y=slice(s[1][0], s[1][1])) for s in slice_indices]
    crs = [tile[0].rio.crs] * len(split_tile)
    del tile
    # extract values and format
    split_tile_values = [s.values for s in split_tile]
    split_tile_values = [np.transpose(s, (1, 2, 0)) for s in split_tile_values] #reogrganize CHW to to HWC
    split_tile_values = [s[:, :, [2, 1, 0]] for s in split_tile_values]  # The image is red in as RGB, Swap to BGRR and B channels,  Assuming the array is in RGB format, reverse the channels to get BGR
    split_tile_values = [np.ascontiguousarray(s) for s in split_tile_values]  # ensure the tile is contiguous 
    return split_tile, split_tile_values, crs


def process_tiles(naip_df, item_dim=640):
    split_tile_list = []
    split_tile_values_list = []
    crs_list = []
    image_name_list = []
    gsd_list = []
    quad_id_list = []
    for index, row in naip_df.iterrows():
        split_tile, split_tile_values, crs = data_processing(row.assets["image"]["href"], item_dim)
        split_tile_list.extend(split_tile)
        split_tile_values_list.extend(split_tile_values)
        crs_list.extend(crs)
        image_name_list.append(row["img_name"]*len(split_tile_values))
        gsd_list.append(row["gsd"]*len(split_tile_values))
        quad_id_list.append(row["quad_id"]*len(split_tile_values))
        
    return pd.DataFrame({"split_tile": split_tile_list, "split_tile_values": split_tile_values_list, 
                         "crs": crs_list,"image_name": image_name_list, "gsd": gsd_list, 
                         "quad_id":quad_id_list})

def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--naip_tile_df", default="/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument('--ntasks', type=int, 
                        help="Specify the number of tasks for the Dask cluster") 
    parser.add_argument('--cpus_per_task', type=int, 
                        help="Specify the number of cpus per task for the Dask cluster") 
    parser.add_argument('--mem_per_cpu', type=str, 
                        help="Specify the memory per cpu for the Dask cluster") 
    args = parser.parse_args()
    return args


def main(args):
    cluster = LocalCluster(n_workers=args.ntasks, threads_per_worker=args.cpus_per_task,
                           memory_limit=args.mem_per_cpu)
    client = Client(cluster)

    naip_ddf = dd.read_parquet(args.naip_tile_df) 
    processed_df = naip_ddf.map_partitions(process_tiles, item_dim=640, 
                                meta={'split_tile': str, "split_tile_values": object, 
                                      "crs_list": str, "image_name": str,
                                      'gsd': float,'quad_id': str})
    #processed_df = processed_df.repartition(partition_size='100MB')
    results = client.persist(processed_df)
    results.to_parquet("/work/csr33/test.parquet", engine="pyarrow") 
    client.close()
    cluster.close()
    
    
if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)