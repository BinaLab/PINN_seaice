# Ignore warning
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import shapefile
import netCDF4
import h5py
import datetime as dt
import pyproj

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers.convolutional import Conv3D
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

###########################################################################################

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
        
        err_u = tf.abs(obs[:, :, :, 0]-prd[:, :, :, 0])
        err_v = tf.abs(obs[:, :, :, 1]-prd[:, :, :, 1])
        err_sic = tf.abs(obs[:, :, :, 2]-prd[:, :, :, 2])
        
        err_sum = tf.sqrt(tf.reduce_mean((err_u + err_v) + err_sic))
        
        # Physical loss term
        u = prd[:, :, :, 0]
        v = prd[:, :, :, 1]
        d_sic = prd[:, 1:-1, 1:-1, 2]
        
        dy = prd[:, 2:, 1:-1, 0] - prd[:, :-2, 1:-1, 0]
        dx = prd[:, 1:-1, 2:, 0] - prd[:, 1:-1, :-2, 0]
        div = dx + dy
        div_std = tf.math.reduce_std(div) / 2
        
        # SIC change
        err_phy = tf.reduce_mean(tf.where((div > 0) & (d_sic > 0), err_u + err_v + err_sic, 0))
        # err_phy = tf.maximum(tf.cast(tf.abs(div_mean)-div_std > 0, tf.float32) * div_mean/tf.abs(div_mean) * d_sic, 0)        
        
        w = tf.constant(1.0)
        
        err_sum += w*err_phy
        return err_sum

###########################################################################################

data_path = "D:\\PINN\\data"

# years = np.array([])
# months = np.array([])
# days = np.array([])
# first = True

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

# lstm_input, lstm_output = make_lstm_input2D(cnn_input, cnn_output, days = 7)
# print("######## TRAINING DATA IS PREPARED (# of samples: {0}) ########".format(len(days)))
# print(np.shape(lstm_input), np.shape(lstm_output))

# days = days[7:]
# months = months[7:]
# years = years[7:]

with open(data_path + '/train_lstm_7days_2019_2021.pkl', 'rb') as file:
    xx, yy, days, months, years, lstm_input, lstm_output = pickle.load(file)
                 
date = 2021

mask1 = (years == date) # Test samples
mask2 = (days % 4 == 2) # Validation samples

test_input = lstm_input[mask1, :, :, :, :]
test_output = lstm_output[mask1, :, :, :]
val_input = lstm_input[(~mask1)&(mask2), :, :, :, :]
val_output = lstm_output[(~mask1)&(mask2), :, :, :]
train_input = lstm_input[(~mask1)&(~mask2), :, :, :, :]
train_output = lstm_output[(~mask1)&(~mask2), :, :, :]
print(np.shape(train_input), np.shape(train_output), np.shape(val_input), np.shape(val_output), np.shape(test_input), np.shape(test_output))

## Convolutional LSTM ####################################################################    
tf.keras.backend.clear_session()
tf.config.experimental.reset_memory_stats('GPU:0')

seq = tf.keras.models.Sequential()

activation = 'linear'
n_filter = 32
seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3),
                   input_shape=np.shape(lstm_input)[1:], activation=activation,
                   padding='same', return_sequences=True))
seq.add(tf.keras.layers.BatchNormalization())

seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3), activation=activation,
                   padding='same', return_sequences=True))
seq.add(tf.keras.layers.BatchNormalization())

seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3), activation=activation,
                   padding='same', return_sequences=True))
seq.add(tf.keras.layers.BatchNormalization())

seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3), activation=activation,
                   padding='same', return_sequences=False))
seq.add(tf.keras.layers.BatchNormalization())

seq.add(layers.Conv2D(n_filter, (3, 3), padding = "same", activation=activation))
seq.add(layers.Conv2D(n_filter, (3, 3), padding = "same", activation=activation))
seq.add(layers.Conv2D(3, (3, 3), padding = "same", activation=activation))

# seq.add(tf.keras.layers.Dense(1))

seq.summary()

n_layers = len(seq.layers)

seq.compile(optimizer='adam', loss=custom_loss())
history = seq.fit(lstm_input, lstm_output, epochs=20, verbose = 1, batch_size=2)
model_name = "convlstm_{0}_{1}_{2}_wo{3}_nophy".format(n_layers, n_filters, activation, str(date).zfill(2))
seq.save("../model/{0}".format(model_name))
history_loss = history.history
with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
    pickle.dump(history_loss, file)
print("Done LSTM without physical loss: {0}".format(model_name))

# seq.compile(optimizer='adam', loss=physics_loss())
# history = seq.fit(lstm_input, lstm_output, epochs=15, verbose = 1, batch_size=10)
# model_name = "convlstm_{0}_{1}_{2}_wo{3}_phy".format(n_layers, n_filters, activation, str(date).zfill(2))
# seq.save("../model/{0}".format(model_name))
# history_loss = history.history
# with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
#     pickle.dump(history_loss, file)
# print("Done LSTM with physical loss: {0}".format(model_name))

tf.keras.backend.clear_session()
tf.config.experimental.reset_memory_stats('GPU:0')
