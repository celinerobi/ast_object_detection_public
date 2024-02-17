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

def download_url(url, destination_folder, destination_filename=None, progress_updater=None, force_download=False): 
    """
    Download a URL to a a file
    Args:
    url(str): url to download
    destination_folder(str): directory to download folder
    destination_filename(str): the name for each of files to download
    return:
    destination_filename
    """

    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is not None:
        destination_filename = os.path.join(destination_folder, destination_filename)
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = os.path.join(destination_folder, url_as_filename)
    if os.path.isfile(destination_filename):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    #  print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))

    return destination_filename

class DownloadProgressBar():
    """
    A progressbar to show the completed percentage and download speed for each image downloaded using urlretrieve.

    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(max_value=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_all_files(gdf, output_dir, connections=6):
    # parse html and retrieve all href urls listed
    # create the pool of worker threads
    with concurrent.futures.ThreadPoolExecutor(connections-4) as executor:
        # dispatch all download tasks to worker threads
        futures = [executor.submit(download_url, url, output_dir, img_name+".tif") \
                   for url, img_name in zip(gdf["img_url"], gdf['img_name'])]
                
        # report results as they become available
        for future in concurrent.futures.as_completed(futures):
            try:
                # retrieve result
                destination_path = future.result()
                print(destination_path)
            except Exception as exc:
                print(str(type(exc)))

        
def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--tile_dir", default="/work/csr33/images_for_predictions/naip_tiles", type=str)
    parser.add_argument('--naip_data_path', type=str, 
                        default="/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet",
                        help='Path to store NAIP information path')
    parser.add_argument("--connections", default=6, type=int)
    parser.add_argument("--chunk_id", default=None, type=int)
    parser.add_argument("--chunked_naip_data_dir", default="/work/csr33/images_for_predictions/chunked_naip_data", type=str)
    parser.add_argument("--chunked_naip_data_filename", default="chunked_naip_data", type=str)


    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.tile_dir, exist_ok=True)
    #os.chdir("/work/csr33/object_detection")
    #make sure processing naip dir exist
    if args.chunk_id is None:
        subset_naip_in_slosh = gpd.read_parquet(args.naip_data_path)
    else:     
        naip_tile_df = f'{args.chunked_naip_data_dir}/{args.chunked_naip_data_filename}_{args.chunk_id}.parquet'
        subset_naip_in_slosh = pd.read_parquet(naip_tile_df) 
        
    #subset_naip_in_slosh["img_url"] = subset_naip_in_slosh.assets.apply(lambda asset: asset["image"]["href"])#
    download_all_files(subset_naip_in_slosh, args.tile_dir, connections=args.connections)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)