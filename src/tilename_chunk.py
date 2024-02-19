import os
import argparse
from glob import glob
import numpy as np



def chunk_df(tile_dir, num_chunks=500):
    array_list = []
    tile_paths = glob(tile_dir + "/*")  # get a list of all of the tiles in tiles directory
    # Calculate the number of rows per chunk
    rows_per_chunk = len(tile_paths) // num_chunks
    df_chunks = [np.array(tile_paths[i : i + rows_per_chunk]) for i in range(0, len(tile_paths), rows_per_chunk)]
    return df_chunks

        
def get_args_parse():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--num_chunks", default=750, type=int)
    parser.add_argument("--tile_dir", default="/work/csr33/images_for_predictions/naip_tiles", type=str)
    parser.add_argument("--tilename_chunks_path", default="/hpc/home/csr33/ast_object_detection/images_for_prediction/tilename_chunks.npz", type=str)
    args = parser.parse_args()
    return args


def main(args):
    df_chunks = chunk_df(args.tile_dir, num_chunks=args.num_chunks)
    # Create a dictionary to hold each array with a unique key
    arrays_dict = {f'{i}': arr for i, arr in enumerate(df_chunks)}
    # Save all arrays into a single .npz file
    np.savez(args.tilename_chunks_path, **arrays_dict)

if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    main(args)