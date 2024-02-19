import geopandas as gpd
import urllib
import time
import urllib.request
import shutil
import os
import argparse
from pathlib import Path
import sys
from zipfile import ZipFile
import math
from contextlib import suppress
from glob import glob
import re

# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import progressbar  
import concurrent
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor


import cv2
import math


def tile_to_chip_array(tile, x, y, item_dim): #used
    """
    https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
    ##
    x: col index
    y: row index
    """
    dimensions = tile.shape[2]
    chip_img = tile[y*item_dim:y*item_dim+item_dim, x*(item_dim):x*(item_dim)+item_dim]
    #add in back space if it is the edge of an image
    if (chip_img.shape[0] != item_dim) & (chip_img.shape[1] != item_dim): #width
        #print("Incorrect Width")
        chip = np.zeros((item_dim,item_dim,dimensions), np.uint8)
        chip[0:chip_img.shape[0], 0:chip_img.shape[1]] = chip_img
        chip_img = chip
    if chip_img.shape[0] != item_dim:  #Height
        black_height = item_dim  - chip_img.shape[0] #Height
        black_width = item_dim #- chip_img.shape[1] #width
        black_img = np.zeros((black_height,black_width,  dimensions), np.uint8)
        chip_img = np.concatenate([chip_img, black_img])
    if chip_img.shape[1] != item_dim: #width
        black_height = item_dim #- chip_img.shape[0] #Height
        black_width = item_dim - chip_img.shape[1] #width
        black_img = np.zeros((black_height,black_width, dimensions), np.uint8)
        chip_img = np.concatenate([chip_img, black_img],1)
    return(chip_img)


def chip_tiles(tile_path, chips_dir, item_dim): 
        """Segment tiles into item_dim x item_dim pixel chips, preserving resolution
        """
        print(tile_path)
        tile_name_wo_ext, ext = os.path.splitext(os.path.basename(tile_path))  # File name
        try:
            tile = cv2.imread(tile_path)
            tile_height, tile_width, tile_channels = tile.shape  # the size of the tile
            # divide the tile into item_dim by item_dim chips (rounding up)
            row_index = math.ceil(tile_height / item_dim)
            col_index = math.ceil(tile_width / item_dim)

            count = 0
            for y in range(0, row_index):
                for x in range(0, col_index):
                    # 
                    # specify the path to save the image
                    chip_img = tile_to_chip_array(tile, x, y, item_dim) #chip tile
                    chip_name = tile_name_wo_ext + '_' + f"{y:02}" + '_' + f"{x:02}" + '.jpg'  #
                    chips_save_path = os.path.join(chips_dir, chip_name)  # row_col.jpg                    
                    cv2.imwrite(os.path.join(chips_save_path), chip_img) # save image
                    count += 1
                    del chip_img
            return tile_name_wo_ext, count
        except Exception as exc:
            print(exc)
        

            


def chip_tiles_concurrent(tile_paths, chips_dir, item_dim=640, connections=6):
    # parse html and retrieve all href urls listed
    # create the pool of worker threads
    with concurrent.futures.ThreadPoolExecutor(connections-4) as executor:
        # dispatch all download tasks to worker threads
        futures = [executor.submit(chip_tiles, tile_path, chips_dir, item_dim=item_dim) for tile_path in tile_paths]
        # report results as they become available
        for future in concurrent.futures.as_completed(futures):
            try:
                # retrieve result
                tile_name_wo_ext, count = future.result()
                print(tile_name_wo_ext, count)
            except Exception as exc:
                print(exc)
                #os.remove(tile)
        
def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--tile_dir", default="/work/csr33/images_for_predictions/naip_tiles", type=str)
    parser.add_argument('--img_dir', type=str, default="/work/csr33/images_for_predictions/naip_imgs")
    parser.add_argument("--connections", default=6, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--chunk_id", type=int)
    parser.add_argument("--tilename_chunks_path", default='/hpc/home/csr33/ast_object_detection/tilename_chunks.npz', type=str)
    args = parser.parse_args()
    return args


def main(args):
    #os.chdir("/work/csr33/object_detection")
    #make sure processing naip dir exist
    start = time.time()
    tile_paths= np.load(args.tilename_chunks_path)[str(args.chunk_id)]

    os.makedirs(args.img_dir, exist_ok=True)
    chip_tiles_concurrent(tile_paths, args.img_dir, item_dim=args.imgsz, connections=args.connections)
    print(time.time() - start)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)