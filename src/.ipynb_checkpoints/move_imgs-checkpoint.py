import argparse
import os
import numpy as np

import shutil

        
def get_args_parse():
    parser = argparse.ArgumentParser("Predict on images")    
    parser.add_argument("--chunk_id",  type=int)
    parser.add_argument("--tilename_chunks_path", default='/hpc/home/csr33/ast_object_detection/images_for_prediction/tilename_chunks.npz', type=str)
    args = parser.parse_args()
    return args


def move(args):
    new_path = "/hpc/group/borsuklab/csr33/object_detection/naip_imgs/"
    naip_imgs = np.load(args.tilename_chunks_path)[str(args.chunk_id)]
    for naip_img in naip_imgs:
        shutil.move(naip_img, new_path)    



if __name__ == '__main__':
    # Get the arguments
    args = get_args_parse()
    print(args)
    move(args)
