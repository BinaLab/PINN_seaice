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

### Set up what month you want to test ===========================
year = 2021
### ==============================================================

n_layers = 10
n_filters = 64

model_name  = "conv2d_{0}_{1}_linear_wo{2}_nophy".format(n_layers, n_filters, str(year).zfill(2)) #, "conv2d_{0}_{1}_linear_wo{2}_phy".format(n_layers, n_filters, str(year).zfill(2))]:
seq = tf.keras.models.load_model("../model/{0}".format(model_name), compile = False)

first = True
data_path = "D:\\PINN\\data"

years = np.array([])
months = np.array([])
days = np.array([])

for month in np.arange(1, 13):
    with open(data_path + '\\train_entire_{0}{1}.pkl'.format(year, str(month).zfill(2)), 'rb') as file:
        xx, yy, input0, output0 = pickle.load(file)
        output0[:, :, :, 2] = output0[:, :, :, 2] - input0[:, :, :, 2]

    n_samples = np.shape(output0)[0]

    cnn_input = input0
    cnn_output = output0
    
    days = np.arange(0, n_samples) + 1
    months = np.ones(n_samples) * month
    years = np.ones(n_samples) * year

    cnn_input, cnn_output = float_to_int(cnn_input, cnn_output)

    print("######## TRAINING DATA IS PREPARED (# of samples: {0}) ########".format(len(days)))
    test_input = cnn_input[:, 41:, :-41, :] / 20000
    test_output = cnn_output[:, 41:, :-41, :] / 20000

    n_samples, row, col, channels = np.shape(test_output)

    pred = seq.predict(test_input)    

    vmax = [10, 10, 100]
    vmin = [-10, -10, 0]
    scaling = [50, 50, 100]
    offset = [0, 0, 0]
    cm = ['Spectral', 'Spectral', 'Blues']

    sic = test_output[:, :, :, 2] + test_input[:, :, :, 2]

    df = pd.DataFrame({})
    fig, ax = plt.subplots(4, 3, figsize = (7,9))

    for c in range(0, channels):

        if c == 2:
            obs = ((test_output[:, :, :, c] + test_input[:, :, :, c]) + offset[c]) *scaling[c] 
            prd = ((pred[:, :, :, c] + test_input[:, :, :, c]) + offset[c]) *scaling[c] 
        else:
            obs = ((test_output[:, :, :, c]) + offset[c]) *scaling[c] 
            prd = ((pred[:, :, :, c]) + offset[c]) *scaling[c] 

        prd[sic == 0] = np.nan
        obs[sic == 0] = np.nan        

        df.loc[c, "MAE{0}".format(c)] = MAE(prd, obs)
        df.loc[c, "R{0}".format(c)] = corr(prd, obs)

        ax[0, c].scatter(prd, obs,  s = 3, alpha = 0.002)
        ax[0, c].text(vmin[c] + (vmax[c]-vmin[c])*0.05, vmax[c] - (vmax[c]-vmin[c])*0.1, "MAE={0:.2f}".format(MAE(obs,prd)))
        ax[0, c].set_xlim(vmin[c], vmax[c])
        ax[0, c].set_ylim(vmin[c], vmax[c])
        ax[0, c].plot([vmin[c], vmax[c]], [vmin[c], vmax[c]], ls = "--", color = "gray")

    #     ax[1, k].pcolormesh(xx, yy, prd, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
    #     ax[2, k].pcolormesh(xx, yy, obs, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])

        ax[1, c].imshow(np.nanmean(prd, axis = 0), vmax = vmax[c], vmin = vmin[c], cmap = cm[c])
        ax[2, c].imshow(np.nanmean(obs, axis = 0), vmax = vmax[c], vmin = vmin[c], cmap = cm[c])
        if c == 2:
            ax[3, c].imshow(np.nanmean(prd-obs, axis = 0), vmax = 50, vmin = -50, cmap = "RdBu")
        else:
            ax[3, c].imshow(np.nanmean(prd-obs, axis = 0), vmax = vmax[c], vmin = vmin[c], cmap = "RdBu")

    df.to_csv("../result/Result_{0}.csv".format(model_name))    
    
    plt.savefig("../result/Test_{0}_{1}{2}.png".format(model_name, year, str(month).zfill(2)), bbox_inches='tight')
    plt.close()



# for date in [2, 5, 8, 11]: #np.arange(1,13):
    
#     with open('../data/train_entire_2021{0}.pkl'.format(str(date).zfill(2)), 'rb') as file:
#         xx, yy, test_input, test_output = pickle.load(file)
#         test_output[:, :, :, 2] = test_output[:, :, :, 2] - test_input[:, :, :, 2]

#     n_layers = 8
#     n_filters = 16

#     for model_name in ["conv2d_10_32_linear_wo{0}_phy".format(str(date).zfill(2)), "conv2d_10_32_linear_wo{0}_nophy".format(str(date).zfill(2))]:
#         seq = tf.keras.models.load_model("../model/{0}".format(model_name), compile = False)

#         pred = seq.predict(test_input)

#         # Cell type visualization -------------------------------
#         for ind in range(0, 25, 15):

#             fig, ax = plt.subplots(4, 3, figsize = (7,9))

#             vmax = [10, 10, 100]
#             vmin = [-10, -10, 0]
#             scaling = [50, 50, 100]
#             offset = [-0.5, -0.5, 0]
#             cm = ['Spectral', 'Spectral', 'Blues']
            
#             sic = test_output[ind, :, :, 2] + test_input[ind, :, :, 2]

#             for k in range(0, 3):
#                 if k == 2:
#                     obs = ((test_output[ind, :, :, k] + test_input[ind, :, :, 2]) + offset[k]) *scaling[k] 
#                     prd = ((pred[ind, :, :, k] + test_input[ind, :, :, 2]) + offset[k]) *scaling[k] 
#                 else:
#                     obs = ((test_output[ind, :, :, k]) + offset[k]) *scaling[k] 
#                     prd = ((pred[ind, :, :, k]) + offset[k]) *scaling[k] 

#                 prd[sic == 0] = np.nan
#                 obs[sic == 0] = np.nan

#                 ax[0, k].scatter(prd, obs,  s = 3, alpha = 0.02)
#                 ax[0, k].text(vmin[k] + (vmax[k]-vmin[k])*0.05, vmax[k] - (vmax[k]-vmin[k])*0.1, "MAE={0:.2f}".format(MAE(obs,prd)))
#                 ax[0, k].set_xlim(vmin[k], vmax[k])
#                 ax[0, k].set_ylim(vmin[k], vmax[k])
#                 ax[0, k].plot([vmin[k], vmax[k]], [vmin[k], vmax[k]], ls = "--", color = "gray")

#             #     ax[1, k].pcolormesh(xx, yy, prd, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
#             #     ax[2, k].pcolormesh(xx, yy, obs, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])

#                 ax[1, k].imshow(prd, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
#                 ax[2, k].imshow(obs, vmax = vmax[k], vmin = vmin[k], cmap = cm[k])
#                 if k == 2:
#                     ax[3, k].imshow(prd-obs, vmax = 50, vmin = -50, cmap = "RdBu")
#                 else:
#                     ax[3, k].imshow(prd-obs, vmax = vmax[k], vmin = vmin[k], cmap = "RdBu")

#             plt.savefig("../result/Test_{0}_{1}_{2}.png".format(str(date).zfill(2), model_name, ind), bbox_inches='tight')
#             plt.close()