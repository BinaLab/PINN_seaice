import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import geopandas
import netCDF4
import h5py
import datetime as dt
import pyproj
from tqdm import tqdm
from pyproj import Proj, transform
from shapely.geometry import Polygon
import cartopy.crs as ccrs

from scipy.interpolate import griddata

import cdsapi
import xarray as xr
from urllib.request import urlopen

import pickle

global data_path
data_path = "D:\\PINN\\data"
# data_path = "C:\\Users\\yok223\\Research\\PINN\\data"

def get_ice_motion(ncfile, i, sampling_size = 1):
# ncfile: input monthly ERA5 file (ncfile)
# field: input variable ('sst', 't2m', 'u10', 'v10')
# bounding_box: processed area (Ross Sea - Amundsen Sea)
# latlon_ib: geocoordinates of the iceberg (lat, lon)
# time_ib: date of the iceberg (datetime format)

    nc = netCDF4.Dataset(ncfile, 'r')
    keys = nc.variables.keys()
    fields = ['u', 'v']

    xs = np.array(nc.variables['x'])[::sampling_size]
    ys = np.array(nc.variables['y'])[::sampling_size]  
    xx, yy = np.meshgrid(xs, ys)
    lat = np.array(nc.variables['latitude'])[::sampling_size, ::sampling_size]
    lon = np.array(nc.variables['longitude'])[::sampling_size, ::sampling_size]

    days = np.array(nc.variables['time']).astype(float)

    for field in fields:                

        data2 = []       

        data = np.array(nc.variables[field][i][::sampling_size, ::sampling_size])
        # cm/s to km/day
        data[data == -9999] = np.nan
        data2.append(data*(3600*24/100000))                        

        data2 = np.array(data2) 
        data_mean = np.array([np.mean(data2, axis = 0)])

        # df[field] = data_mean.flatten()

        if field == "u":
            u = data2 # data_mean
            # u[np.isnan(u)] = 0
        elif field == "v":
            v = data2 # data_mean
            # v[np.isnan(v)] = 0
    
    nc.close()
    
    return xx, yy, lat, lon, u, v


def get_SIC(t1, xx, yy):
    ## Read SIC data ==================================================
    h5file = data_path + "/SIC/AMSR_U2_L3_SeaIce25km_B04_{0}.he5".format(dt.datetime.strftime(t1, "%Y%m%d"))
    
    if os.path.exists(h5file):
        f = h5py.File(h5file)

        lat2 = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lat'][:]
        lon2 = f['HDFEOS']['GRIDS']['NpPolarGrid25km']['lon'][:]
        sic = f['/HDFEOS/GRIDS/NpPolarGrid25km/Data Fields/SI_25km_NH_ICECON_DAY'][:].astype(float)
        sic[sic <= 0] = 0
        sic[sic > 100] = 0

        # EPSG:4326 (WGS84); EPSG:3408 (NSIDC EASE-Grid North - Polar pathfinder sea ice movement)
        # ESPG:3411 (NSIDC Sea Ice Polar Stereographic North - SIC data)
        inProj = Proj('epsg:4326')  
        outProj = Proj('epsg:3408')
        xx2,yy2 = transform(inProj,outProj,lat2,lon2)
        grid_sic = griddata((xx2.flatten(), yy2.flatten()), sic.flatten(), (xx, yy), method='linear')
        grid_sic[np.isnan(grid_sic)] = 0
        return grid_sic * 0.01  # Change into 0-1
    
    else:
        print("Filename is NOT correct!")

def retrieve_ERA5(year):
    c = cdsapi.Client()
    # dataset to read
    dataset = 'reanalysis-era5-single-levels'
    # flag to download data
    download_flag = False
    # api parameters 
    params = {
        'format': 'netcdf',
        'product_type': 'reanalysis',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature'],
        'year':[str(year)],
        'month': ['01', '02', '03', '04', '05', '06','07', '08', '09','10', '11', '12'],
        'day': ['01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
               ],
        'area': [90, -180, 50,180],
        'time': ['12:00'],
        'grid': [1, 0.5],
        'area': [90, -180, 50, 180],
        }

    # retrieves the path to the file
    fl = c.retrieve(dataset, params)

    # load into memory
    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())

    return ds

def rotate_vector(u, v, lon):
    angle = lon*np.pi/180 # rotation angle (radian)
    u2 = u*np.cos(angle) - v*np.sin(angle)
    v2 = u*np.sin(angle) + v*np.cos(angle)
    return u2, v2

def get_ERA5(ds, i, xx, yy):
    lat3, lon3 = np.meshgrid(ds.latitude, ds.longitude)
    inProj = Proj('epsg:4326')  
    outProj = Proj('epsg:3408')
    xx3,yy3 = transform(inProj,outProj,lat3,lon3)
    t2m = np.array(ds.t2m[i]).transpose()
    u10 = np.array(ds.u10[i]).transpose()
    v10 = np.array(ds.v10[i]).transpose()
    
    u10, v10 = rotate_vector(u10, v10, lon3)
    
    grid_t2m = griddata((xx3.flatten(), yy3.flatten()), np.array(t2m).flatten(), (xx, yy), method='linear')
    grid_u10 = griddata((xx3.flatten(), yy3.flatten()), np.array(u10).flatten(), (xx, yy), method='linear')
    grid_v10 = griddata((xx3.flatten(), yy3.flatten()), np.array(v10).flatten(), (xx, yy), method='linear')
    
    grid_t2m[np.isnan(grid_t2m)] = 0
    grid_u10[np.isnan(grid_u10)] = 0
    grid_v10[np.isnan(grid_v10)] = 0
    
    return grid_t2m, grid_u10, grid_v10

def make_dataset(year, n_samples, ds, w = 1, datatype = "cell"):
    # ncfile = glob.glob("F:\\2022_Ross\\ERA5\\icemotion_daily_sh_25km_{0}*.nc".format(year))[0]
    ncfile = data_path + "/Sea_ice_drift/icemotion_daily_nh_25km_{0}0101_{0}1231_v4.1.nc".format(year)
    nc = netCDF4.Dataset(ncfile, 'r')
    ## Adjust the number of training datasets ===========================
    days = np.array(nc.variables['time']).astype(float)[:]
    row, col = np.shape(np.array(nc.variables['latitude']))
    
    # Initialize grid input ==========================================
    grid_input = np.zeros([len(n_samples), row, col, 6])
    grid_output = np.zeros([len(n_samples), row, col, 3])
    
    first = True
    
    for i, idx in tqdm(enumerate(n_samples)):
        t1 = dt.datetime(1970, 1, 1) + dt.timedelta(days = days[idx])
        t2 = dt.datetime(1970, 1, 1) + dt.timedelta(days = days[idx]+1)  

        ## Read ice motion data ===========================================
        sampling_size = 1
        xx, yy, lat, lon, u, v = get_ice_motion(ncfile, idx, sampling_size)
        grid_u = np.mean(u, axis = 0)
        grid_v = np.mean(v, axis = 0)      

        ## Read SIC data ==================================================
        grid_sic = get_SIC(t1, xx, yy)

        ## Read ERA5 data =================================================
        grid_t2m, grid_u10, grid_v10 = get_ERA5(ds, idx, xx, yy)

        grid_input[i, :, :, 0] = grid_u / 50 + 0.5
        grid_input[i, :, :, 1] = grid_v / 50 + 0.5
        grid_input[i, :, :, 2] = grid_sic
        grid_input[i, :, :, 3] = (grid_t2m - 240)/(320 - 240) #Max temp = 320 K, Min temp = 240 K)
        grid_input[i, :, :, 4] = grid_u10 / 50 + 0.5
        grid_input[i, :, :, 5] = grid_v10 / 50 + 0.5

        _, _, _, _, u2, v2 = get_ice_motion(ncfile, idx+1, sampling_size)
        grid_u2 = np.mean(u2, axis = 0)
        grid_v2 = np.mean(v2, axis = 0) 
        grid_output[i, :, :, 0] = grid_u2 / 50 + 0.5
        grid_output[i, :, :, 1] = grid_v2 / 50 + 0.5
        grid_sic2 = get_SIC(t2, xx, yy)
        grid_output[i, :, :, 2] = grid_sic2
        
        # Masking ======================================
        mask1 = (np.isnan(grid_u))
        mask2 = (np.isnan(grid_u2))

        if datatype == "cell":
            xx1, yy1 = [], []
            for m in range(w, row-w):
                for n in range(w, col-w):
                    ip = np.array([grid_input[i, m-w:m+w+1, n-w:n+w+1, :]])
                    if mask1[m,n] == False: #np.prod(ip[0, :, :, 2]) > 0:
                        op = np.array([grid_output[i, m-w:m+w+1, n-w:n+w+1, :]])
                        xx1.append(xx[m, n])
                        yy1.append(yy[m, n])
                        if first:
                            conv_input = ip
                            conv_output = op
                            first = False
                        else:
                            conv_input = np.concatenate((conv_input, ip), axis = 0)
                            conv_output = np.concatenate((conv_output, op), axis = 0)            

        elif datatype == "entire":
            var_ip = np.shape(grid_input)[3]
            var_op = np.shape(grid_output)[3]
            
            conv_input = np.copy(grid_input)
            conv_output = np.copy(grid_output)
            
            for m in range(0, var_ip):
                subset = grid_input[i, :, :, m]
                subset[mask1] = 0
                conv_input[i, :, :, m] = subset
                
            for n in range(0, var_op):
                subset = grid_output[i, :, :, n]
                subset[mask2] = 0
                conv_output[i, :, :, n] = subset
                
            xx1, yy1 = xx, yy

        elif datatype == "table":
            
            xx1, yy1 = [], []
            for m in range(w, row-w):
                for n in range(w, col-w):
                    ip = np.array([grid_input[i, m-w:m+w+1, n-w:n+w+1, :].flatten()])
                    if np.prod(grid_sic[m-w:m+w+1, n-w:n+w+1]) > 0:
                        op = np.array([grid_output[i, m-w:m+w+1, n-w:n+w+1, :].flatten()])
                        xx1.append(xx[m, n])
                        yy1.append(yy[m, n])
                        
                        if first:
                            conv_input = ip
                            conv_output = op
                            first = False
                        else:
                            conv_input = np.concatenate((conv_input, ip), axis = 0)
                            conv_output = np.concatenate((conv_output, op), axis = 0)  

    return xx1, yy1, conv_input, conv_output

def MAE(obs, prd):
    return np.nanmean(abs(obs-prd))