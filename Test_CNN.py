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

for date in np.arange(1,13):
    
    with open('../data/train_entire_2021{0}.pkl'.format(str(date).zfill(2)), 'rb') as file:
        xx, yy, test_input, test_output = pickle.load(file)
        test_output[:, :, :, 2] = test_output[:, :, :, 2] - test_input[:, :, :, 2]

    n_layers = 8
    n_filters = 16

    for model_name in ["conv2d_10_32_linear_phy", "conv2d_10_32_linear_nophy"]:
        seq = tf.keras.models.load_model("../model/{0}".format(model_name), compile = False)

        pred = seq.predict(test_input)

        # Cell type visualization -------------------------------
        for ind in range(0, 25, 5):

            fig, ax = plt.subplots(4, 3, figsize = (7,9))

            vmax = [10, 10, 100]
            vmin = [-10, -10, 0]
            scaling = [50, 50, 100]
            offset = [-0.5, -0.5, 0]
            cm = ['Spectral', 'Spectral', 'Blues']
            
            sic = test_output[ind, :, :, 2] + test_input[ind, :, :, 2]

            for k in range(0, 3):
                if k == 2:
                    obs = ((test_output[ind, :, :, k] + test_input[ind, :, :, 2]) + offset[k]) *scaling[k] 
                    prd = ((pred[ind, :, :, k] + test_input[ind, :, :, 2]) + offset[k]) *scaling[k] 
                else:
                    obs = ((test_output[ind, :, :, k]) + offset[k]) *scaling[k] 
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

            plt.savefig("../result/Test_{0}_{1}_{2}.png".format(str(date).zfill(2), model_name, ind), bbox_inches='tight')
            plt.close()