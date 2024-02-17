import geopandas as gpd
import urllib
import time
import urllib.request
import shutil
import os
import argparse
from pathlib import Path
from glob import glob

# Less standard, but still pip- or conda-installable
import pandas as pd
import numpy as np
import progressbar  
import concurrent
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor



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
    parser = argparse.ArgumentParser("download all tiles in naip slosh modeled area")
    parser.add_argument("--tile_dir", default="/work/csr33/images_for_predictions/naip_tiles", type=str)
    parser.add_argument('--naip_data_path', type=str, 
                        default="/work/csr33/images_for_predictions/naip_tile_in_slosh_modeled_area.parquet",
                        help='Path to store NAIP information path')
    parser.add_argument("--connections", default=6, type=int)
    args = parser.parse_args()
    return args


def main(args):    
    # list of tiles with errors
    tile_paths_error = ["/work/csr33/images_for_predictions/naip_tiles/m_4307154_se_19_060_20211024.tif",
                        "/work/csr33/images_for_predictions/naip_tiles/m_3008742_sw_16_030_20211113.tif",
                        "/work/csr33/images_for_predictions/naip_tiles/m_2608104_ne_17_060_20191129_20200113.tif",
                        "/work/csr33/images_for_predictions/naip_tiles/m_4107357_sw_18_060_20211105.tif",
                        "/work/csr33/images_for_predictions/naip_tiles/m_3008741_se_16_030_20211113.tif",
                        
                        '/work/csr33/images_for_predictions/naip_tiles/m_3607610_sw_18_060_20210827.tif',
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008802_nw_16_030_20211124.tif',
                        '/work/csr33/images_for_predictions/naip_tiles/m_3009018_sw_15_030_20211119.tif',
                        '/work/csr33/images_for_predictions/naip_tiles/m_3009234_nw_15_030_20211221.tif',
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008736_sw_16_030_20211113.tif',
                        
                        '/work/csr33/images_for_predictions/naip_tiles/m_3407851_se_17_060_20200922.tif',
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008831_nw_16_030_20211119.tif',
                       '/work/csr33/images_for_predictions/naip_tiles/m_3009245_se_15_030_20211222.tif',
                       '/work/csr33/images_for_predictions/naip_tiles/m_3009046_sw_15_030_20211115.tif',
                       '/work/csr33/images_for_predictions/naip_tiles/m_3008814_nw_16_030_20211112.tif']
                        
                       #'/work/csr33/images_for_predictions/naip_tiles/m_2608104_sw_17_060_20191129_20200113.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3009255_nw_15_030_20211212.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3009037_nw_15_030_20211115.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3108858_sw_16_030_20211124.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3009154_sw_15_030_20211116.tif',
                        
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3009149_nw_15_030_20211113.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_4406720_sw_19_060_20210907.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3008831_ne_16_030_20211119.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_2909032_se_15_030_20211115.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3009258_nw_15_030_20211221.tif',
                       #'/work/csr33/images_for_predictions/naip_tiles/m_3009264_nw_15_030_20211212.tif',
                       #/work/csr33/images_for_predictions/naip_tiles/m_3009245_se_15_030_20211222.tif]
                    """
                        '/work/csr33/images_for_predictions/naip_tiles/m_3009238_nw_15_030_20211222.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008948_ne_16_030_20211114.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008808_nw_16_030_20211120.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3907441_sw_18_060_20190726_20191209.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3009245_ne_15_030_20211222.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008737_se_16_030_20211113.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008818_ne_16_030_20211119.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008903_nw_16_030_20211115.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3707741_ne_18_060_20210628.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008717_sw_16_030_20211119.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3707740_nw_18_060_20210912.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008814_ne_16_030_20211112.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3009045_ne_15_030_20211115.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_2708018_sw_17_060_20191129_20200113.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3009257_se_15_030_20211221.tif'
                        '/work/csr33/images_for_predictions/naip_tiles/m_3008839_sw_16_030_20211113.tif'
                    """
    #remove error tiles
    for tile_path in tile_paths_error:
        if os.path.exists(tile_path):
            os.remove(tile_path)
    #identify list of tiles (given that they have all been downloaded)
    naip_tile_in_slosh_modeled_area = gpd.read_parquet(args.naip_data_path)

    # identify the downloaded tile paths
    tile_paths = glob(args.tile_dir + "/*")  # get a list of all of the tiles in tiles directory
    #tile_names = [os.path.splitext(tile_name)[0] for tile_name in tile_names]
    tile_names = [os.path.splitext(os.path.basename(tile_path))[0] for tile_path in tile_paths]
    #identify the info for tanks that need to be downloaded again
    remaining_download = naip_tile_in_slosh_modeled_area[~naip_tile_in_slosh_modeled_area.img_name.isin(tile_names)]
    print(len(remaining_download))
    # download remaining
    download_all_files(remaining_download, args.tile_dir, connections=args.connections)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)