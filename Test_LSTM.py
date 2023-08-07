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

# files = glob.glob('../data/train_entire_20*.pkl')
# print(files)

# first = True
# for f in files:
#     with open(f, 'rb') as file:
#         xx, yy, input0, output0 = pickle.load(file)
#         output0[:, :, :, 2] = output0[:, :, :, 2] - input0[:, :, :, 2]
#     if first:
#         train_input = input0
#         train_output = output0
#         first = False
#     else:
#         train_input = np.concatenate((train_input, input0))
#         train_output = np.concatenate((train_output, output0))
    
with open('../data/train_entire_202103.pkl', 'rb') as file:
    xx, yy, test_input, test_output = pickle.load(file)
    test_output[:, :, :, 2] = test_output[:, :, :, 2] - test_input[:, :, :, 2]
    


# lstm_input, lstm_output = make_lstm_input2D(train_input, train_output, days = 7)
lstm_test_input, lstm_test_output = make_lstm_input2D(test_input, test_output, days = 7)
# lstm_input = lstm_input / 255.
# lstm_output = lstm_output / 255.
lstm_test_input = lstm_test_input / 255.
lstm_test_output = lstm_test_output / 255.
print(np.shape(lstm_test_input), np.shape(lstm_test_output))

model_name = "conv_lstm_6_32_linear_phy"
seq = tf.keras.models.load_model("../model/{0}".format(model_name), compile = False)

pred = seq.predict(lstm_test_input)

# Cell type visualization -------------------------------
for ind in range(0, 25, 5):

    fig, ax = plt.subplots(4, 3, figsize = (7,9))

    vmax = [10, 10, 100]
    vmin = [-10, -10, 0]
    scaling = [50, 50, 100]
    offset = [-0.5, -0.5, 0]
    cm = ['Spectral', 'Spectral', 'Blues']
    
    sic = lstm_test_output[ind, :, :, 2] + lstm_test_input[ind, 6, :, :, 2]

    for k in range(0, 3):
        if k == 2:
            obs = ((lstm_test_output[ind, :, :, k] + lstm_test_input[ind, 6, :, :, 2]) + offset[k]) *scaling[k] 
            prd = ((pred[ind, :, :, k] + lstm_test_input[ind, 6, :, :, 2]) + offset[k]) *scaling[k] 
        else:
            obs = ((lstm_test_output[ind, :, :, k]) + offset[k]) *scaling[k] 
            prd = ((pred[ind, :, :, k]) + offset[k]) *scaling[k] 
        
        prd[sic == 0] = np.nan
        obs[sic == 0] = np.nan

        ax[0, k].scatter(prd, obs,  s = 3, alpha = 0.02)
        ax[0, k].text(vmin[k] + (vmax[k]-vmin[k])*0.05, vmax[k] - (vmax[k]-vmin[k])*0.1, "MAE={0:.2f}".format(MAE(obs,prd)))
        ax[0, k].set_xlim(vmin[k], vmax[k])
        ax[0, k].set_ylim(vmin[k], vmax[k])
        ax[0, k].plot([vmin[k], vmax[k]], [vmin[k], vmax[k]], ls = "--", color = "gray")

    #     ax[1, k].pcolormesh(xx, yy, prd, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
    #     ax[2, k].pcolormesh(xx, yy, obs, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])

        ax[1, k].imshow(prd, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
        ax[2, k].imshow(obs, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
        if k == 2:
            ax[3, k].imshow(prd-obs, vmax = 50, vmin = -50, cmap = "RdBu")
        else:
            ax[3, k].imshow(prd-obs, vmax = vmax[k], vmin = vmin[k], cmap = "RdBu")

    plt.savefig("../result/Test_{0}_{1}.png".format(model_name, ind), bbox_inches='tight')
    print("Test_{0}_{1}.png".format(model_name, ind))