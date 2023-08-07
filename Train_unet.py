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
        div = dx + dy
        
#         div_std = tf.math.reduce_std(div)
#         sic_std = tf.math.reduce_std(err_sic)
        
        # SIC change
        err_phy = tf.reduce_mean(tf.where((div > 0) & (d_sic > 0), err_u + err_v + err_sic, 0))
        # err_phy = tf.maximum(tf.cast(tf.abs(div_mean)-div_std > 0, tf.float32) * div_mean/tf.abs(div_mean) * d_sic, 0)        
        
        w = tf.constant(1.0)
        
        err_sum += w*err_phy
        return err_sum
    
    
##########################################################################################3

years = np.array([])
months = np.array([])
days = np.array([])
first = True

data_path = "D:\\PINN\\data"

for year in [2019, 2020, 2021]:
    
    for month in np.arange(1,13):

        with open(data_path + '\\train_entire_{0}{1}.pkl'.format(year, str(month).zfill(2)), 'rb') as file:
            xx, yy, input0, output0 = pickle.load(file)
            output0[:, :, :, 2] = output0[:, :, :, 2] - input0[:, :, :, 2]

        n_samples = np.shape(output0)[0]

        if first:
            cnn_input = input0
            cnn_output = output0
            first = False
        else:
            cnn_input = np.concatenate((cnn_input, input0))
            cnn_output = np.concatenate((cnn_output, output0))

        days = np.concatenate((days, np.arange(0, n_samples) + 1))
        months = np.concatenate((months, np.ones(n_samples) * month))
        years = np.concatenate((years, np.ones(n_samples) * year))

cnn_input, cnn_output = float_to_int(cnn_input, cnn_output)
cnn_input = cnn_input / 20000
cnn_output = cnn_output / 20000

print("######## TRAINING DATA IS PREPARED ########")
print(cnn_input.dtype, cnn_output.dtype)
                  
for date in [2019]: #np.arange(10,13):
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
    print(train_input.dtype, train_output.dtype)

    ## Design CNN architecture =========================================================
    model = models.Sequential()

    n_layers = 10
    n_filters = 32
    n_epochs = 30
    activation = "tanh"

    inputs = tf.keras.layers.Input(np.shape(train_input[0, :, :, :]))
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)


    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(
        tf.keras.layers.UpSampling2D(size = (2,2))(conv5))
    merge6 = tf.keras.layers.concatenate([conv4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(
        tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(
        tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(
        tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(3, 3, activation = activation, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv10 = tf.keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.Model(inputs = inputs, outputs = conv9)
    model.summary()

    ## Normal CNN (without physical loss) ==========================================================
    model.compile(optimizer='adam', loss=custom_loss())
    history = model.fit(train_input, train_output, epochs=n_epochs, validation_data=(val_input, val_output), verbose = 2, batch_size=8)
    model_name = "unet_{0}_{1}_{2}_wo{3}_nophy".format(n_layers, n_filters, activation, str(date).zfill(2))
    model.save("../model/{0}".format(model_name))
    with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
        pickle.dump(history.history, file)
    print("Done CNN without physical loss: {0}".format(model_name))

    # if os.path.exists("../result/{0}".format(model_name)):
    #     pass
    # else:
    #     os.mkdir("../result/{0}".format(model_name))

#     pred = model.predict(test_input)
#     n_samples, row, col, channels = np.shape(test_output)
#     df = pd.DataFrame({})

#     scaling = [50, 50, 100]
#     offset = [-0.5, -0.5, 0]

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
#     model.compile(optimizer='adam', loss=physics_loss())
#     history = model.fit(train_input, train_output, epochs=n_epochs, validation_data=(val_input, val_output), verbose = 2, batch_size=8)
#     model_name = "unet_{0}_{1}_{2}_wo{3}_phy".format(n_layers, n_filters, activation, str(date).zfill(2))
#     model.save("../model/{0}".format(model_name))
#     with open('../model/history_{0}.pkl'.format(model_name), 'wb') as file:
#         pickle.dump(history.history, file)
#     print("Done CNN with physical loss: {0}".format(model_name))

#     pred = model.predict(test_input)
#     n_samples, row, col, channels = np.shape(test_output)
#     df = pd.DataFrame({})

#     scaling = [50, 50, 100]
#     offset = [-0.5, -0.5, 0]

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

#     del model, df, test_input, test_output, train_input, train_output

    tf.keras.backend.clear_session()
    tf.config.experimental.reset_memory_stats('GPU:0')