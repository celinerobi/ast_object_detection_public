import os
import argparse
import math

import pandas as pd
import numpy as np

import rioxarray

def tile_to_chip_array(chip_img, item_dim=640, bands=3): #used
    """
    ##
    """
    #add in back space if it is the edge of an image
    if (chip_img.shape[0] != item_dim) & (chip_img.shape[1] != item_dim): #width
        #print("Incorrect Width")
        chip = np.zeros((item_dim, item_dim, bands), np.uint8)
        chip[0:chip_img.shape[0], 0:chip_img.shape[1]] = chip_img
        chip_img = chip
    if chip_img.shape[0] != item_dim:  #Height
        black_height = item_dim  - chip_img.shape[0] #Height
        black_width = item_dim #- chip_img.shape[1] #width
        black_img = np.zeros((black_height, black_width, bands), np.uint8)
        chip_img = np.concatenate([chip_img, black_img])
    if chip_img.shape[1] != item_dim: #width
        black_height = item_dim #- chip_img.shape[0] #Height
        black_width = item_dim - chip_img.shape[1] #width
        black_img = np.zeros((black_height, black_width, bands), np.uint8)
        chip_img = np.concatenate([chip_img, black_img],1)
    return(chip_img)

def data_processing(row, item_dim=640):
    #read in image
    tile = rioxarray.open_rasterio(row["img_url"])#.squeeze()
    tile_channels, tile_height, tile_width = tile.shape  # the size of the tile
    # divide the tile into item_dim by item_dim chips (rounding up)
    row_index = math.ceil(tile_height / item_dim)
    col_index = math.ceil(tile_width / item_dim)   
    # determine slices for tile
    slice_indices = [[[x*item_dim, x*item_dim+item_dim], [y*item_dim, y*item_dim+item_dim]] for y in range(row_index) for x in range(col_index)]
    num_of_splits = len(slice_indices)
    #split tile data array
    split_tile = [tile.isel(x=slice(s[0][0], s[0][1]), y=slice(s[1][0], s[1][1])) for s in slice_indices]
    crs = [tile[0].rio.crs.to_string()] * num_of_splits
    x_coords = [tile.coords['x'].values.tolist() for tile in split_tile]
    y_coords = [tile.coords['y'].values.tolist() for tile in split_tile]
    
    # extract values and format
    split_tile_values = [s.values.tolist() for s in split_tile]
    split_tile_values = [np.transpose(s, (1, 2, 0)) for s in split_tile_values] #reogrganize CHW to to HWC
    split_tile_values = [s[:, :, [2, 1, 0]] for s in split_tile_values]  # The image is red in as RGB, Swap to BGRR and B channels,  Assuming the array is in RGB format, reverse the channels to get BGR
    split_tile_values = [tile_to_chip_array(chip_img, item_dim=item_dim, bands=3) for chip_img in split_tile_values]
    split_tile_values = [s.astype(np.uint8) for chip_img in split_tile_values]
    split_tile_values = [np.ascontiguousarray(s).flatten().tolist() for s in split_tile_values]  # ensure the tile is contiguous 
    
    
    del tile, split_tile
   
    #dtypes = {"y_coords": object, "x_coords": object, "split_tile_values": object,
    #            "crs": str, "image_name":str, "quad_id":str, "gsd": float}
    
    return pd.DataFrame({"y_coords": y_coords, "x_coords": x_coords, "split_tile_values": split_tile_values, 
                         "crs": crs, "image_name": [row["img_name"]]*num_of_splits,
                         "quad_id": [row["quad_id"]]*num_of_splits,
                         "gsd": [row["gsd"]]*num_of_splits})#,  dtype=dtypes)


def data_processing_over_chunk(df_chunk, args):
    file_path = os.path.join(args.processing_naip_dir, f"{args.processing_naip_filename}_{args.chunk_id}.parquet")
    
    for i, (_, row) in enumerate(df_chunk.iterrows()):
        temp_processed_naip = data_processing(row, args.imgsz)
        if i == 0:
            temp_processed_naip.to_parquet(file_path, engine='fastparquet')
        else:
            temp_processed_naip.to_parquet(file_path, engine='fastparquet', append=True)
        del temp_processed_naip

        
def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--naip_tile_df", default="/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet", type=str)
    parser.add_argument("--processing_naip_dir", default="/work/csr33/images_for_predictions/processed_naip_data", type=str)
    parser.add_argument("--processing_naip_filename", default="processed_naip_data", type=str)
    parser.add_argument("--imgsz", default=640, type=int)
    args = parser.parse_args()
    return args



def main(args):
    #os.chdir("/work/csr33/object_detection")
    #make sure processing naip dir exist
    os.makedirs(args.processing_naip_dir, exist_ok=True)
    #determine chunk-number
    file_name_wo_ext = os.path.splitext(os.path.basename(args.naip_tile_df))[0]
    args.chunk_id = file_name_wo_ext.rsplit("_",1)[1]

    naip_df = pd.read_parquet(args.naip_tile_df) 
    naip_df["img_url"] = naip_df.assets.apply(lambda asset: asset["image"]["href"])#
    data_processing_over_chunk(naip_df, args)
    
    #for df_chunk in chunk_dataframe(naip_df, chunksize=1000):
    #    data_processing_over_chunk(df_chunk, args)
    #    print(chunk)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)