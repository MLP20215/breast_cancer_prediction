from datetime import datetime
import random
import os
import sys

import numpy as np
import h5py
import json

from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt

from preprocessing.datamodel import SlideManager
from preprocessing.processing import split_negative_slide, split_positive_slide, create_tumor_mask, rgb2gray
from preprocessing.util import TileMap

from generate_tiles import project_path 
from generate_tiles import camelyon17_path 
from generate_tiles import generated_data_path 
from generate_tiles import mgr 
from generate_tiles import n_slides
from generate_tiles import level 
from generate_tiles import tile_size 
from generate_tiles import poi 
from generate_tiles import poi_tumor 
from generate_tiles import overlap_tumor 
from generate_tiles import overlap 
from generate_tiles import max_tiles_per_slide 


with open("annotation_status.json", 'r') as f:
    annotation_status = json.load(f)



single_file = '{}/all_wsis_{}x{}_poi{}_poiTumor{}_level{}.hdf5'.format(generated_data_path, tile_size, tile_size, 
                                                       poi, poi_tumor, level)
h5_single = h5py.File(single_file, 'w')

for f in os.listdir(generated_data_path):
    if f.startswith('patient'):
        filename = os.path.join(generated_data_path, f)
        with h5py.File(filename, 'r') as h5:
            for key in h5.keys():
                print('processing: "{}", shape: {}'.format(key, h5[key].shape))
                if h5[key].shape[0] > 0: ### dont create dsets for WSIs with 0 tiles
                    if key in annotation_status['positive_slides']:
                        dset = h5_single.create_dataset(key + "_tumor", 
                            h5[key].shape, 
                            dtype=np.uint8,
                            data=h5[key][:],
                            compression=0)
                    else:
                        dset = h5_single.create_dataset(key + "_normal", 
                            h5[key].shape, 
                            dtype=np.uint8,
                            data=h5[key][:],
                            compression=0)

            
h5_single.close()
