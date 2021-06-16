import numpy as np
import random
import os
import sys
import h5py
import math
from datetime import datetime

from matplotlib import pyplot as plt
from preprocessing.util import find_files
from preprocessing.datamodel import SlideManager
from preprocessing.util import TileMap

import tensorflow as tf
from tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'

# Set up hdf5 file path
generated_data_path = '../data_generated/'
hdf5_file_path = os.path.join(generated_data_path, "all_wsis_256x256_poi0.2_poiTumor0.6_level3.hdf5")

# Save checkpoints for our model
model_checkpoint = os.path.join(generated_data_path, "model_checkpoint.ckpt")
model_final = os.path.join(generated_data_path, "model_final.hdf5")

data_file_path = h5py.File(hdf5_file_path,'r',libver='latest',swmr=True)
#print('List of data entires')
#for key in data_file_path.keys():
#    print(key, 'with shape', data_file_path[key].shape)

class TissueDataset():
    """Data set for preprocessed WSIs of the CAMELYON17 data set."""
        
    def __init__(self, path, percentage=.5, first_part=True):      
        self.h5_file = path
        self.h5 = h5py.File(path, 'r', libver='latest', swmr=True)
        self.perc = percentage
        self.first_part = first_part
        self.dataset_names = list(self.h5.keys())
        self.neg = [i for i in self.dataset_names if 'ormal' in i]
        self.pos = [i for i in self.dataset_names if 'umor' in i]
        self.dims = self.h5[self.neg[0]][0].shape
    
    def __get_tiles_from_path(self, dataset_names, max_wsis, number_tiles):
        tiles = np.ndarray((number_tiles, 256, 256, 3))
        for i in range(number_tiles):
            file_idx = np.random.randint(0, max_wsis)
            dset = self.h5[dataset_names[file_idx]]
            len_ds = len(dset)
            max_tiles = math.ceil(len_ds * self.perc)
            if self.first_part:
                rnd_idx = np.random.randint(0, max_tiles)
            else:
                rnd_idx = np.random.randint(len_ds - max_tiles, len_ds)
            ### crop random 256x256
            if self.dims[1] > 256:
                rand_height = np.random.randint(0, self.dims[0]-256)
                rand_width = np.random.randint(0, self.dims[1]-256)
            else:
                rand_height = 0
                rand_width = 0
            tiles[i] = dset[rnd_idx,rand_height:rand_height+256,rand_width:rand_width+256]
        tiles = tiles / 255.
        return tiles
    
    def __get_random_positive_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.pos, len(self.pos), number_tiles), np.ones((number_tiles))
    
    def __get_random_negative_tiles(self, number_tiles):
        return self.__get_tiles_from_path(self.neg, len(self.neg), number_tiles), np.zeros((number_tiles))
    
    def generator(self, num_neg=10, num_pos=10, data_augm=False, mean=[0.,0.,0.], std=[1.,1.,1.]):
        while True:
            x, y = self.get_batch(num_neg, num_pos, data_augm)
            for i in [0,1,2]:
                x[:,:,:,i] = (x[:,:,:,i] - mean[i]) / std[i]
            yield x, y

    def get_batch(self, num_neg=10, num_pos=10, data_augm=False):
        x_p, y_p = self.__get_random_positive_tiles(num_pos)
        x_n, y_n = self.__get_random_negative_tiles(num_neg)
        x = np.concatenate((x_p, x_n), axis=0)
        y = np.concatenate((y_p, y_n), axis=0)
        if data_augm:
            ### some data augmentation mirroring / rotation
            if np.random.randint(0,2): x = np.flip(x, axis=1)
            if np.random.randint(0,2): x = np.flip(x, axis=2)
            x = np.rot90(m=x, k=np.random.randint(0,4), axes=(1,2))
        ### randomly arrange in order
        p = np.random.permutation(len(y))
        return x[p], y[p]
        
train_data = TissueDataset(path=hdf5_file_path,  percentage=0.5, first_part=True)
val_data = TissueDataset(path=hdf5_file_path, percentage=0.5, first_part=False)

x, y = train_data.get_batch(num_neg=3, num_pos=3)
print(x.shape)
print(y)

plt.figure(figsize=(12,4))

itera = train_data.generator(num_neg=1, num_pos=1, data_augm=True)
for x, y in itera:
    print(x.shape)
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{} - class {}'.format(i, y[i]))
        ax.imshow(x[i])
        ax.axis('off') 
    break # generate yields infinite random samples, so we stop after first

mean_pixel = np.full((3),0.0)
std_pixel = np.full((3),1.0)

mean_img = np.full((256,256,3),0.0)
std_img = np.full((256,256,3),0.0)
mean_img[:,:] = mean_pixel
std_img[:,:] = std_pixel

plt.figure(figsize=(12,4))

# Mean image
ax = plt.subplot(1, 2, 1)
plt.tight_layout()
ax.set_title('mean image')
ax.imshow(mean_img)
ax.axis('off')
# Std Deviation image
ax = plt.subplot(1, 2, 2)
plt.tight_layout()
ax.set_title('std deviation image')
ax.imshow(std_img)
ax.axis('off') 

print('Mean colors: ', mean_pixel)
print('Std Dev colors: ', std_pixel)

plt.figure(figsize=(12,4))

itera = train_data.generator(1, 1, True, mean_pixel, std_pixel)
for x, y in itera:
    print(x.shape)
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{} - class {}'.format(i, y[i]))
        ax.imshow(x[i])
        ax.axis('off') 
    break # generate yields infinite random samples, so we stop after first

base_model = keras.applications.InceptionResNetV2(
                                 include_top=False, 
                                 weights='imagenet', 
                                 input_shape=(256,256,3), 
                                 )

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
predictions = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

# If you interrupted your training, load the checkpoint here:
#model.load_weights(model_checkpoint)

train_accs = []
train_losses = []
val_accs = []
val_losses = []

batch_size_neg=10
batch_size_pos=10
batches_per_train_epoch = 50
batches_per_val_epoch = 25
epochs = 25

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_checkpoint, 
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

now1 = datetime.now()

### Uncomment to easily disable color normalization
#mean_pixel = [0.,0.,0.] 
#std_pixel = [1.,1.,1.]  

for i in range(epochs):
    hist = model.fit(
            train_data.generator(batch_size_neg, batch_size_pos, True, mean_pixel, std_pixel),
            steps_per_epoch=batches_per_train_epoch, 
            validation_data=val_data.generator(batch_size_neg, batch_size_pos, False, mean_pixel, std_pixel),
            validation_steps=batches_per_val_epoch,
            callbacks=[cp_callback], workers=1, use_multiprocessing=False, max_queue_size=10)
    
    train_accs.append(hist.history['accuracy'])
    train_losses.append(hist.history['loss'])
    val_accs.append(hist.history['val_accuracy'])
    val_losses.append(hist.history['val_loss'])

now2 = datetime.now()
print(now2 - now1)

# Save entire model to a HDF5 file
model.save(model_final)
