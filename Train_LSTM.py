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
import metpy.calc as mpcalc
from metpy.units import units

import cdsapi
import xarray as xr
from urllib.request import urlopen

import pickle

###########################################################################################

def make_lstm_input2D(data_input, data_output, days = 7):
    # Input & output should be entire images for CNN
    n_samples, row, col, var_ip = np.shape(data_input)
    _, _, _, var_op = np.shape(data_output)
    row,col = 320, 320;
    lstm_input = np.zeros([n_samples-days, days, row, col, var_ip], dtype="int")
    lstm_output = np.zeros([n_samples-days, row, col, var_op], dtype="int")
    
    for n in range(0, n_samples-days):
        for i in range(0, days):
            for v in range(0, var_ip):
                lstm_input[n, i, :, :, v] = (data_input[n+i, 41:, :-41, v]*255).astype(int)
            for v in range(0, var_op):
                lstm_output[n, :, :, v] = (data_output[n+days, 41:, :-41, v]*255).astype(int)
    return lstm_input, lstm_output

def MAE(obs, prd):
    return np.nanmean(abs(obs-prd))

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
        err_phy = tf.reduce_mean(tf.where((div > 0) & (d_sic > 0), div * d_sic, 0))
        # err_phy = tf.maximum(tf.cast(tf.abs(div_mean)-div_std > 0, tf.float32) * div_mean/tf.abs(div_mean) * d_sic, 0)        
        
        w = tf.constant(2.0)
        
        err_sum += w*err_phy
        return err_sum

###########################################################################################

files = glob.glob('../data/train_entire_20*.pkl')
print(files)

first = True
for f in files:
    with open(f, 'rb') as file:
        xx, yy, input0, output0 = pickle.load(file)
        output0[:, :, :, 2] = output0[:, :, :, 2] - input0[:, :, :, 2]
    if first:
        train_input = input0
        train_output = output0
        first = False
    else:
        train_input = np.concatenate((train_input, input0))
        train_output = np.concatenate((train_output, output0))
    
with open('../data/train_entire_202103.pkl', 'rb') as file:
    xx, yy, test_input, test_output = pickle.load(file)
    test_output[:, :, :, 2] = test_output[:, :, :, 2] - test_input[:, :, :, 2]


lstm_input, lstm_output = make_lstm_input2D(train_input, train_output, days = 7)
lstm_test_input, lstm_test_output = make_lstm_input2D(test_input, test_output, days = 7)
lstm_input = lstm_input / 255.
lstm_output = lstm_output / 255.
print(np.shape(lstm_input), np.shape(lstm_output))

# # Plot input & output ------------------------------------------------------- 
# ind = 300
# fig, ax = plt.subplots(7, 6, figsize = (10, 10))
# for i in range(0, np.shape(ax)[0]):
#     for k in range(0, np.shape(ax)[1]):
#         p = ax[i, k].imshow(lstm_input[ind, i, :, :, k])
# #         fig.colorbar(p, ax=ax[i, k], shrink = 0.5)
#         # ax[2, k].scatter(xx, yy, c=test_output[:, 1, 1, k]*scaling[k], s=1, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
    
# fig, ax = plt.subplots(1, 3, figsize = (15,2))
# for k in range(0, len(ax)):
#     p = ax[k].imshow(lstm_output[ind, :, :, k])
#     fig.colorbar(p, ax=ax[k], shrink = 0.5)
#     # ax[2, k].scatter(xx, yy, c=test_output[:, 1, 1, k]*scaling[k], s=1, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])


## Convolutional LSTM ####################################################################    
seq = tf.keras.models.Sequential()

activation = 'linear'
n_filter = 32
seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3),
                   input_shape=np.shape(lstm_input)[1:], activation=activation,
                   padding='same', return_sequences=False))
seq.add(tf.keras.layers.BatchNormalization())

# seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3), activation=activation,
#                    padding='same', return_sequences=True))
# seq.add(tf.keras.layers.BatchNormalization())

# seq.add(tf.keras.layers.ConvLSTM2D(filters=n_filter, kernel_size=(3, 3), activation=activation,
#                    padding='same', return_sequences=False))
# seq.add(tf.keras.layers.BatchNormalization())

seq.add(layers.Conv2D(n_filter, (3, 3), padding = "same", activation=activation))
seq.add(layers.Conv2D(n_filter, (3, 3), padding = "same", activation=activation))
seq.add(layers.Conv2D(n_filter, (3, 3), padding = "same", activation=activation))
seq.add(layers.Conv2D(3, (3, 3), padding = "same", activation=activation))

# seq.add(tf.keras.layers.ConvLSTM2D(filters=3, kernel_size=(1, 1), activation=activation,
#                    padding='same', return_sequences=False))
# seq.add(tf.keras.layers.BatchNormalization())

# seq.add(tf.keras.layers.Conv3D(filters=3, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))

# seq.add(tf.keras.layers.Dense(1))

seq.summary()

n_layers = len(seq.layers)

seq.compile(optimizer='adam', loss=custom_loss())
history = seq.fit(lstm_input, lstm_output, epochs=15, verbose = 1, batch_size=10)
model_name = "conv_lstm_{0}_{1}_{2}_nophy".format(n_layers, n_filter, activation)
seq.save("../model/{0}".format(model_name))
with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
    pickle.dump(history.history, file)
print("Done LSTM without physical loss: {0}".format(model_name))

seq.compile(optimizer='adam', loss=physics_loss())
history = seq.fit(lstm_input, lstm_output, epochs=15, verbose = 1, batch_size=10)
model_name = "conv_lstm_{0}_{1}_{2}_phy".format(n_layers, n_filter, activation)
seq.save("../model/{0}".format(model_name))
with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
    pickle.dump(history.history, file)
print("Done LSTM with physical loss: {0}".format(model_name))
