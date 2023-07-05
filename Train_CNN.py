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

################################################################################

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
    
##########################################################################################3



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
    
## Normal CNN (without physical loss)
model = models.Sequential()

n_layers = 10
n_filters = 32
activation = "linear"

model.add(layers.Conv2D(n_filters, (3, 3), padding = "same", activation=activation, input_shape=np.shape(train_input)[1:]))

for i in range(0, n_layers-1):
    model.add(layers.Conv2D(n_filters, (3, 3), padding = "same", activation=activation))

model.add(layers.Conv2D(3, (3, 3), padding = "same", activation=activation))
model.summary()

model.compile(optimizer='adam', loss=custom_loss())
history = model.fit(train_input, train_output, epochs=30, verbose = 1, batch_size=32)
model_name = "conv2d_{0}_{1}_{2}_nophy".format(n_layers, n_filters, activation)
model.save("../model/{0}".format(model_name))
with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
    pickle.dump(history.history, file)
print("Done LSTM without physical loss: {0}".format(model_name))

model.compile(optimizer='adam', loss=physics_loss())
history = model.fit(train_input, train_output, epochs=30, verbose = 1, batch_size=32)
model_name = "conv2d_{0}_{1}_{2}_phy".format(n_layers, n_filters, activation)
model.save("../model/{0}".format(model_name))
with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
    pickle.dump(history.history, file)
print("Done LSTM without physical loss: {0}".format(model_name))