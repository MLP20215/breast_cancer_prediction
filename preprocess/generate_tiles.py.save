from datetime import datetime
import random
import os
import sys

import numpy as np
import h5py

from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt

from preprocessing.datamodel import SlideManager
from preprocessing.processing import split_negative_slide, split_positive_slide, create_tumor_mask, rgb2gray
from preprocessing.util import TileMap


# Set up project & data path
project_path = "/home/MLP20215/project"
camelyon17_path = os.path.join(project_path, "CAMELYON17")
generated_data_path = os.path.join(project_path, "data_generated")
 
mgr = SlideManager(cam17_dir=camelyon17_path)
n_slides= len(mgr.slides)

# Zoom level 3 & 256x256 pixels for each tile
level = 3
tile_size = 256

poi = 0.20 # 20% of negative tiles must contain tissue (in contrast to slide background)
poi_tumor = 0.60 # 60% of pos tiles must contain metastases
# to not have too few positive tile, we use half overlapping tilesize
overlap_tumor = tile_size // 2
# we have enough normal tissue, so negative tiles will be less of a problem
overlap = 0.0
max_tiles_per_slide = 1000


# Variables for test


if __name__ == '__main__':
    tiles_pos = 0
    tiles_neg = 0

    # Slides with cancer (i.e. annotated slides)
    for i in range(len(mgr.annotated_slides)):
        try: 
            
            filename = '{}/{}_{}x{}_poi{}_poiTumor{}_level{}.hdf5'.format(generated_data_path, mgr.annotated_slides[i].name, tile_size, tile_size, 
                                                           poi, poi_tumor, level)
            # 'w-' creates file, fails if exists
            h5 = h5py.File(filename, "w-", libver='latest')
            
            # create a new and unconsumed tile iterator
            tile_iter = split_positive_slide(mgr.annotated_slides[i], level=level,
                                             tile_size=tile_size, overlap=overlap_tumor,
                                             poi_threshold=poi_tumor) 

            tiles_batch = []
            for tile, bounds in tile_iter:
                if len(tiles_batch) % 10 == 0: print('positive slide #:', i, 'tiles so far:', len(tiles_batch))
                if len(tiles_batch) > max_tiles_per_slide: break
                tiles_batch.append(tile)

            # creating a date set in the file
            dset = h5.create_dataset(mgr.annotated_slides[i].name, 
                                     (len(tiles_batch), tile_size, tile_size, 3), 
                                     dtype=np.uint8,
                                     data=np.array(tiles_batch),
                                     compression=0)   
            h5.close()

            tiles_pos += len(tiles_batch)
            print(datetime.now(), i, '/', len(mgr.annotated_slides), '  tiles  ', len(tiles_batch))
            print('pos tiles total: ', tiles_pos)

        except:
            print('slide nr {}/{} failed'.format(i, len(mgr.annotated_slides)))
            print(sys.exc_info()[0])


    # Slides with no cancer (i.e. negative slides)
    for i in range(len(mgr.negative_slides)): 
        try:
            filename = '{}/{}_{}x{}_poi{}_poiTumor{}_level{}.hdf5'.format(generated_data_path, mgr.negative_slides[i].name, tile_size, tile_size, 
                                                           poi, poi_tumor, level)
            # 'w-' creates file, fails if exists
            h5 = h5py.File(filename, "w-", libver='latest')
            
            # load the slide into numpy array
            arr = np.asarray(mgr.negative_slides[i].get_full_slide(level=4))

            # convert it to gray scale
            arr_gray = rgb2gray(arr)

            # calculate otsu threshold
            threshold = threshold_otsu(arr_gray)

            # create a new and unconsumed tile iterator
            # because we have so many  negative slides we do not use overlap
            tile_iter = split_negative_slide(mgr.negative_slides[i], level=level,
                                             otsu_threshold=threshold,
                                             tile_size=tile_size, overlap=overlap,
                                             poi_threshold=poi)

            tiles_batch = []
            for tile, bounds in tile_iter:
                if len(tiles_batch) % 10 == 0: print('neg slide:', i, 'tiles so far:', len(tiles_batch))
                if len(tiles_batch) > max_tiles_per_slide: break
                tiles_batch.append(tile)

            # creating a date set in the file
            dset = h5.create_dataset(mgr.negative_slides[i].name, 
                                     (len(tiles_batch), tile_size, tile_size, 3), 
                                     dtype=np.uint8,
                                     data=np.array(tiles_batch),
                                     compression=0)
            h5.close()
            
            tiles_neg += len(tiles_batch)
            print(datetime.now(), i, '/', len(mgr.negative_slides), '  tiles  ', len(tiles_batch))
            print('neg tiles total: ', tiles_neg)
            
        except:
            print('slide nr {}/{} failed'.format(i, len(mgr.negative_slides)))
            print(sys.exc_info()[0])
