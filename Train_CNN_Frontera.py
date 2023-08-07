# Ignore warning
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import netCDF4
import h5py
import datetime as dt
import pyproj

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# from keras.layers.convolutional import Conv3D
# from keras.backend.tensorflow_backend import set_session
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tqdm import tqdm

from pyproj import Proj, transform
from shapely.geometry import Polygon
import cartopy.crs as ccrs

from scipy.interpolate import griddata

import cdsapi
import xarray as xr
from urllib.request import urlopen

import pickle

from functions import *

################################################################################

class custom_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, obs, prd):
        # err_u = tf.sqrt(tf.reduce_mean((obs[:, 1, 1, 0]-prd[:, 1, 1, 0])**2)) / tf.reduce_mean(tf.abs(obs[:, 1, 1, 0]))
        # err_v = tf.sqrt(tf.reduce_mean((obs[:, 1, 1, 1]-prd[:, 1, 1, 1])**2)) / tf.reduce_mean(tf.abs(obs[:, 1, 1, 1]))
        # err_sic = tf.sqrt(tf.reduce_mean((obs[:, 1, 1, 2]-prd[:, 1, 1, 2])**2)) / tf.reduce_mean(obs[:, 1, 1, 2]) * 0.01
        
        obs = tf.cast(obs, tf.float32)
        prd = tf.cast(prd, tf.float32)
        
        err_u = tf.abs(obs[:, :, :, 0]-prd[:, :, :, 0])
        err_v = tf.abs(obs[:, :, :, 1]-prd[:, :, :, 1])
        err_sic = tf.abs(obs[:, :, :, 2]-prd[:, :, :, 2])
        
        err_sum = tf.reduce_mean((err_u + err_v) + err_sic)
        # err_sum = tf.sqrt(tf.reduce_mean(err_u*err_sic)) + tf.sqrt(tf.reduce_mean(err_v*err_sic))
        return err_sum
    
class physics_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, obs, prd):
        # err_u = tf.sqrt(tf.reduce_mean((obs[:, 1, 1, 0]-prd[:, 1, 1, 0])**2)) / tf.reduce_mean(tf.abs(obs[:, 1, 1, 0]))
        # err_v = tf.sqrt(tf.reduce_mean((obs[:, 1, 1, 1]-prd[:, 1, 1, 1])**2)) / tf.reduce_mean(tf.abs(obs[:, 1, 1, 1]))
        # err_sic = tf.sqrt(tf.reduce_mean((obs[:, 1, 1, 2]-prd[:, 1, 1, 2])**2)) / tf.reduce_mean(obs[:, 1, 1, 2]) * 0.01
        
        obs = tf.cast(obs, tf.float32)
        prd = tf.cast(prd, tf.float32)
        
        err_u = tf.abs(obs[:, 1:-1, 1:-1, 0]-prd[:, 1:-1, 1:-1, 0])
        err_v = tf.abs(obs[:, 1:-1, 1:-1, 1]-prd[:, 1:-1, 1:-1, 1])
        err_sic = tf.abs(obs[:, 1:-1, 1:-1, 2]-prd[:, 1:-1, 1:-1, 2])
        
        err_sum = tf.sqrt(tf.reduce_mean((err_u + err_v) + err_sic))
        
        # Physical loss term
        u = prd[:, :, :, 0]
        v = prd[:, :, :, 1]
        d_sic = prd[:, 1:-1, 1:-1, 2]
        
        dy = prd[:, 2:, 1:-1, 0] - prd[:, :-2, 1:-1, 0]
        dx = prd[:, 1:-1, 2:, 0] - prd[:, 1:-1, :-2, 0]
        div = dx/50 + dy/50
        
#         div_std = tf.math.reduce_std(div)
#         sic_std = tf.math.reduce_std(err_sic)
        
        # SIC change
        # err_phy = tf.reduce_mean(tf.where((div > 0) & (d_sic > 0), div * err_sic, 0))
        err_phy = tf.reduce_mean(tf.where((div > 0) & (d_sic > 0), err_u + err_v + err_sic, 0))
        # err_phy = tf.maximum(tf.cast(tf.abs(div_mean)-div_std > 0, tf.float32) * div_mean/tf.abs(div_mean) * d_sic, 0)        
        
        w = tf.constant(1.0)
        
        err_sum += w*err_phy
        return err_sum
    
##########################################################################################3

# years = np.array([])
# months = np.array([])
# days = np.array([])
# first = True

data_path = ""

# for year in [2019, 2020, 2021]:
    
#     for month in np.arange(1,13):

#         with open(data_path + '\\train_entire_{0}{1}.pkl'.format(year, str(month).zfill(2)), 'rb') as file:
#             xx, yy, input0, output0 = pickle.load(file)
#             output0[:, :, :, 2] = output0[:, :, :, 2] - input0[:, :, :, 2]

#         n_samples = np.shape(output0)[0]

#         if first:
#             cnn_input = input0
#             cnn_output = output0
#             first = False
#         else:
#             cnn_input = np.concatenate((cnn_input, input0))
#             cnn_output = np.concatenate((cnn_output, output0))

#         days = np.concatenate((days, np.arange(0, n_samples) + 1))
#         months = np.concatenate((months, np.ones(n_samples) * month))
#         years = np.concatenate((years, np.ones(n_samples) * year))

# cnn_input, cnn_output = float_to_int(cnn_input, cnn_output)
# cnn_input = cnn_input / 20000
# cnn_output = cnn_output / 20000

with open(data_path + '/train_cnn_2019_2021.pkl', 'rb') as file:
    xx, yy, days, months, years, cnn_input, cnn_output = pickle.load(file)

print(cnn_input.dtype, cnn_output.dtype)
print("######## TRAINING DATA IS PREPARED (# of samples: {0}) ########".format(len(days)))

#### Tensorflow setting #######
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)
###############################

# Create a TensorFlow Timeline object
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Run your TensorFlow model
with tf.Session() as sess:
    # Run your TensorFlow model with profiling enabled
    sess.run(
        model,
        options=run_options,
        run_metadata=run_metadata)

# Save the Timeline data to a file
with open('timeline.json', 'w') as f:
    f.write(
        timeline.Timeline(
            run_metadata.step_stats).generate_chrome_trace_format())
                  
date = 2021
mask1 = (years == date) # Test samples
mask2 = (days % 4 == 2) # Validation samples

test_input = cnn_input[mask1, 41:, :-41, :]
test_output = cnn_output[mask1, 41:, :-41, :]
val_input = cnn_input[(~mask1)&(mask2), 41:, :-41, :]
val_output = cnn_output[(~mask1)&(mask2), 41:, :-41, :]
train_input = cnn_input[(~mask1)&(~mask2), 41:, :-41, :]
train_output = cnn_output[(~mask1)&(~mask2), 41:, :-41, :]

del mask1, mask2

print(np.shape(train_input), np.shape(train_output), np.shape(val_input), np.shape(val_output), np.shape(test_input), np.shape(test_output))

## Design CNN architecture =========================================================
model = models.Sequential()

n_layers = 8
n_filters = 16
n_epochs = 30
win = 5
activation = "linear"

model.add(layers.Conv2D(n_filters, (win, win), padding = "same", activation=activation, input_shape=np.shape(train_input)[1:]))

for i in range(0, n_layers-1):
    model.add(layers.Conv2D(n_filters, (win, win), padding = "same", activation=activation))

model.add(layers.Conv2D(3, (win, win), padding = "same", activation=activation))
model.summary()

## Normal CNN (without physical loss) ==========================================================
# model.compile(optimizer='adam', loss=custom_loss())
# history = model.fit(train_input, train_output, epochs=n_epochs, validation_data=(val_input, val_output), verbose = 2, batch_size=16)
# model_name = "conv2d_{0}_{1}_{2}_wo{3}_nophy".format(n_layers, n_filters, activation, str(date).zfill(2))
# model.save("../model/{0}".format(model_name))
# with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
#     pickle.dump(history.history, file)
# print("Done CNN without physical loss: {0}".format(model_name))

# if os.path.exists("../result/{0}".format(model_name)):
#     pass
# else:
#     os.mkdir("../result/{0}".format(model_name))

#     pred = model.predict(test_input)
#     n_samples, row, col, channels = np.shape(test_output)
#     df = pd.DataFrame({})

#     scaling = [50, 50, 100]
#     offset = [0, 0, 0]

#     for k in range(0, n_samples):
#         sic = (test_output[k, :, :, 2] + test_input[k, :, :, 2])
#         for c in range(0, channels):
#             obs = ((test_output[k, :, :, c]) + offset[c]) *scaling[c] 
#             prd = ((pred[k, :, :, c]) + offset[c]) *scaling[c] 

#             prd[sic == 0] = np.nan
#             obs[sic == 0] = np.nan        

#             df.loc[k, "MAE{0}".format(c)] = MAE(prd, obs)
#             df.loc[k, "R{0}".format(c)] = corr(prd, obs)
#             del obs, prd

#     df.to_csv("../result/Result_{0}.csv".format(model_name)) 

## Normal CNN (with physical loss) ==========================================================
model.compile(optimizer='adam', loss=physics_loss())
history = model.fit(train_input, train_output, epochs=n_epochs, validation_data=(val_input, val_output), verbose = 0, batch_size=16)
model_name = "conv2d_{0}_{1}_{2}_wo{3}_phy".format(n_layers, n_filters, activation, str(date).zfill(2))
model.save("{0}".format(model_name))
with open('history_{0}.pkl'.format(model_name), 'wb') as file:
    pickle.dump(history.history, file)
print("Done CNN with physical loss: {0}".format(model_name))
