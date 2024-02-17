GDAL_HTTP_MAX_RETRY=10
GDAL_HTTP_RETRY_DELAY=30

import os
import argparse
import math
import time
import pandas as pd
import numpy as np

import rioxarray


def split_raster(tile, args):
    tile_channels, tile_height, tile_width = tile.shape  # the size of the tile
    # divide the tile into item_dim by item_dim chips (rounding up)
    row_index = math.ceil(tile_height / args.imgsz)
    col_index = math.ceil(tile_width / args.imgsz)   
    # determine slices for tile
    slice_indices = [[[x*args.imgsz, x*args.imgsz+args.imgsz],
                      [y*args.imgsz, y*args.imgsz+args.imgsz]] for y in range(row_index) for x in range(col_index)]
    num_of_splits = len(slice_indices)
    return slice_indices, num_of_splits




def split_format_array(tile, slice_indices):
    # extract values and format
    split_tile = [tile.isel(x=slice(s[0][0], s[0][1]), y=slice(s[1][0], s[1][1])) for s in slice_indices]
    split_tile_values = [np.transpose(chip_img.values.tolist(), (1, 2, 0)) for chip_img in split_tile] #reogrganize CHW to to HWC
    split_tile_values = [chip_img[:, :, [2, 1, 0]] for chip_img in split_tile_values]  # The image is red in as RGB, Swap to BGRR and B channels,  Assuming the array is in RGB format, reverse the channels to get BGR
    split_tile_values = [tile_to_chip_array(chip_img, item_dim = args.imgsz, bands=3) for chip_img in split_tile_values]
    split_tile_values = [np.ascontiguousarray(chip_img.astype(np.uint8)).flatten().tolist() for chip_img in split_tile_values]  # ensure the tile is contiguous 
    return split_tile, split_tile_values

def split_format_array(tile, slice_indices):
    # extract values and format
    return split_tile, split_tile_values

def data_processing(row, args):
    tile = rioxarray.open_rasterio(row['img_url'])
    #split tile data array
    slice_indices, num_of_splits = split_raster(tile, args)
    split_tile, split_tile_values = split_format_array(tile, slice_indices)

    #get tile geo information
    crs = [tile[0].rio.crs.to_string()] * num_of_splits
    x_coords = [tile.coords['x'].values.tolist() for tile in split_tile]
    y_coords = [tile.coords['y'].values.tolist() for tile in split_tile]
    del tile, split_tile
    
    return pd.DataFrame({"x_coords": x_coords, "y_coords": y_coords, "split_tile_values": split_tile_values, 
                         "crs": crs, "image_name": [row["img_name"]]*num_of_splits,
                         "quad_id": [row["quad_id"]]*num_of_splits, "gsd": [row["gsd"]]*num_of_splits})#,  dtype=dtypes)


def data_processing_handle_errors(row, retry, args):
    
    try:
        df = data_processing(row, args)
        return df
    except Exception as err:
        print(type(err))    # the exception type
        time.sleep(args.backoff_factor * (2 ** retry))
        return None
        
        
def data_processing_retry(row, args):
    for retry in range(args.max_retries):
        print(retry)
        df = data_processing_handle_errors(row, retry, args)
        if df is not None:
            break
    return df

        
def data_processing_over_chunk(df_chunk, args):
    file_path = os.path.join(args.processing_naip_dir, f"{args.processing_naip_filename}_{args.chunk_id}.parquet")
    # remove file if it exists 
    if os.path.exists(file_path):
        os.remove(file_path)
    
    for idx, (_, row) in enumerate(df_chunk.iterrows()):
        start_time = time.time()
        
        temp_processed_naip = data_processing_retry(row, args)
        if temp_processed_naip is None:
            continue
        if os.path.exists(file_path):
            temp_processed_naip.to_parquet(file_path, engine='fastparquet', append=True)
        else:
            temp_processed_naip.to_parquet(file_path, engine='fastparquet')
            
        del temp_processed_naip
        end_time = time.time()
        execution_time = end_time - start_time
        print(idx, "Execution time:", execution_time, "seconds")

        
def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--chunked_naip_data_dir", default="/work/csr33/images_for_predictions/chunked_naip_data", type=str)
    parser.add_argument("--chunked_naip_data_filename", default="chunked_naip_data", type=str)
    parser.add_argument("--chunk_id",  type=int)

    
    parser.add_argument("--backoff_factor", default=10, type=float)
    parser.add_argument("--max_retries", default=10, type=int)
    
    parser.add_argument("--processing_naip_dir", default="/work/csr33/images_for_predictions/processed_naip_data", type=str)
    parser.add_argument("--processing_naip_filename", default="processed_naip_data", type=str)
    parser.add_argument("--imgsz", default=640, type=int)

    args = parser.parse_args()
    return args


def main(args):
    #os.chdir("/work/csr33/object_detection")
    #make sure processing naip dir exist

    os.makedirs(args.processing_naip_dir, exist_ok=True)
    
    naip_tile_df = f'{args.chunked_naip_data_dir}/{args.chunked_naip_data_filename}_{args.chunk_id}.parquet'
    naip_df = pd.read_parquet(naip_tile_df) 
    data_processing_over_chunk(naip_df, args)
    

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)