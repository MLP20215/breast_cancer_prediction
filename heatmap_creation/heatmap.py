import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import math
from skimage.filters import threshold_otsu
from skimage.transform import resize

from preprocessing.datamodel import SlideManager
from preprocessing.processing import split_negative_slide, split_positive_slide, create_tumor_mask, rgb2gray, create_otsu_mask_by_threshold
from preprocessing.util import TileMap


def heatmap(slide, model):
    slide_name = slide.name
    print("processing {}".format(slide_name))
    # Annotation mask
    mask = create_tumor_mask(slide, level=8)
    
    # Prediction
    size = (int(slide.dimensions[0] / 512), int(slide.dimensions[1] / 512))
    arr = np.asarray(slide.get_full_slide(level=6))
    arr_gray = rgb2gray(arr)
    threshold = threshold_otsu(arr_gray)
    tile_iter = split_negative_slide(
        slide, level=3,
        otsu_threshold=threshold,  # otsu threshold calculated earlier
        tile_size=256,
        overlap=0,                 # no overlap
        poi_threshold=0.9
    )
    pred = np.zeros(size)
    for tile, bound in tqdm(tile_iter):
        prob = model.predict(tile.reshape(1, 256, 256, 3) / 255)
        for i in range(4):
            for j in range(4):
                pred[int(bound[0][0] / 512) + i, int(bound[0][1] / 512) + j] = prob[0][0]
    print("prediction done.")

    plt.figure()
    plt.title("True")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mask, cmap='gray', interpolation='spline16')
    plt.savefig("{}_heatmap_true.png".format(slide_name), facecolor='w', dpi=600)

    plt.figure()
    plt.title("Pred")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pred.T, cmap='gray', interpolation='spline16')
    plt.savefig("{}_heatmap_pred.png".format(slide_name), facecolor='w', dpi=600)

    np.save("{}_mask_true.npy".format(slide_name), mask)
    np.save("{}_mask_pred.npy".format(slide_name), pred.T)
# Set up project & data path
project_path = "../"
camelyon16_path = os.path.join(project_path, "CAMELYON16")
camelyon17_path = os.path.join(project_path, "CAMELYON17")
generated_data_path = os.path.join(project_path, "data_generated")
model_path = os.path.join(generated_data_path, "model_final.hdf5")
heatmap_path = os.path.join(generated_data_path, "test_set_predictions")


# Load model
model = tf.keras.models.load_model(model_path)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Zoom level 3 & 256x256 pixels for each tile
level = 3
tile_size = 256
overlap = tile_size // 2
poi = 20

# predict with CAMELYON16 Data
mgr = SlideManager(cam16_dir=camelyon16_path)

# Modify this for custom test data
slides = mgr.slides[-6:-3]

for slide in slides:
    heatmap(slide, model)
