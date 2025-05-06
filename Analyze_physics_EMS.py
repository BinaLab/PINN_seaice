### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import numpy.ma as ma
import math
from datetime import datetime
from tqdm import tqdm
import time
import pickle

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import glob, os
 
# from torch.utils.tensorboard import SummaryWriter

from torch_model import *

import argparse
import os 

def MAE(prd, obs):
    return np.nanmean(abs(obs-prd))

def MAE_grid(prd, obs):
    err = abs(obs-prd)
    return np.nanmean(err, axis=0)

def RMSE(prd, obs):
    err = np.square(obs-prd)
    return np.nanmean(err)**0.5

def RMSE_grid(prd, obs):
    err = np.square(obs-prd)
    return np.nanmean(err, axis=0)**0.5

def corr(prd, obs):
    prd = prd.flatten()
    obs = obs.flatten()    
    r = ma.corrcoef(ma.masked_invalid(prd), ma.masked_invalid(obs))[0, 1]
    return r

def corr_grid(prd, obs):
    r1 = np.nansum((prd-np.nanmean(prd))*(obs-np.nanmean(obs)),axis=0)
    r2 = np.nansum(np.square(prd-np.nanmean(prd)), axis=0)*np.nansum(np.square(obs-np.nanmean(obs)),axis=0)
    r = r1/r2**0.5
    return r

def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch Example')

    parser.add_argument(
        '--model-dir',
        default='../model',
        help='Model directory',
    )
    parser.add_argument(
        '--data-dir',
        default='../data',
        help='Data directory',
    )
    parser.add_argument(
        '--data-file',
        default="/train_cnn_2009_2015_v7.pkl",
        help='Data file',
    )
    parser.add_argument(
        '--result-dir',
        default='../result_phy',
        help='Result directory',
    )
    parser.add_argument(
        '--no-cuda',
        # action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--phy',
        type=str,
        default='phy',
        help='filename of dataset',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default="hisunet",
        help='types of the neural network model (e.g. unet, cnn, fc)',
    )
    
    parser.add_argument(
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )

    # Set automatically by torch distributed launch
    parser.add_argument(
        '--local-rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )
    
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main() -> None:

    args = parse_args()

    ### Read data
    model_path = args.model_dir #"D:\\PINN\\model"
    data_path = args.data_dir # "D:\\PINN\\data"
    result_path = args.result_dir # "D:\\PINN\\results_phy"
    data_file = args.data_file
    
    phy = args.phy    
    model_type = args.model_type
    if args.no_cuda:
        device = torch.device('cpu')
        device_name = 'cpu'
    else:
        device = torch.device('cuda')
        device_name = 'gpu'
    
    with open(data_path + data_file, 'rb') as file:
        xx, yy, days, months, years, data0, target0 = pickle.load(file)  
    
    data0 = data0[:, :, :, [0,1,2,3,4,5]]
    target0 = target0[:,:,:,:-1]
    
    data0 = torch.tensor(data0, dtype=torch.float32)
    target0 = torch.tensor(target0, dtype=torch.float32)
    pred0 = target0.clone()
    
    with open(data_path + "/landmask_256.pkl", 'rb') as file:
        landmask = pickle.load(file)

    year = int(data_file[-11:-7])
    row, col = 256, 256

    print(f"#### DATA IS LOADED: {year} ####")    
    
    for ratio in [0.2, 0.5, 1.0]:
        for phy_w in [0, 0.2, 1.0, 5.0]: #[0, 0.2, 0.5, 1.0, 2.0, 5.0]:        
            for sat_w in [0, 0.2, 1.0, 5.0]: #[0, 0.2, 0.5, 1.0, 2.0, 5.0]:

                df = {}
                rmse_total = {}
                r_total = {} 
                rmse_all_mat = {}
    
                torch.cuda.empty_cache()

                if phy_w == 0 and sat_w == 0:
                    phy = "nophy"
                else:
                    phy = "phy"
    
                in_channels, out_channels = 18, 3
                model = HIS_UNet(in_channels, out_channels, landmask, row, 3, phy)
                if phy_w == 0 and sat_w == 0:
                    gpu = 6
                else:
                    gpu = 4
                # res = f"{result_dir}\\test_torch_hisunet_satv7_all_2016_2022_r{ratio}_pw{pw}_{phy}_d3f1_gpu8_{year}.pkl"
                if year == 2015:
                    model_name = model_path + f"/torch_{model_type}_satv7_all_2016_2022_r{ratio}_pw{phy_w}_sw{sat_w}_d3f1_gpu{gpu}.pth"
                elif year == 2022:
                    model_name = model_path + f"/torch_{model_type}_satv7_all_2009_2015_r{ratio}_pw{phy_w}_sw{sat_w}_d3f1_gpu{gpu}.pth"
    
                keyname = f"{model_type}_{ratio}_{phy_w}_{sat_w}"
                print(keyname)
    
                device = "cpu"
                if device == "cuda":
                    model = nn.DataParallel(model)
                    model.load_state_dict(torch.load(model_name, map_location=device))
                    model = model.to(device)
                elif device == "cpu":
                    model = nn.DataParallel(model)
                    model.load_state_dict(torch.load(model_name, map_location=device))
                    model = model.module.to(device)
    
                # sic_obs = target[:, 2, :, :]*100
                sic_max = np.nanmax(target0[:, 2, :, :], axis=0)
    
                id_start = 0
                umonths = np.unique(months)
                uyears = np.unique(years)
    
                mdf = pd.DataFrame({'year': np.ones(len(umonths))*year})
    
                rmse_all = np.zeros([out_channels, len(uyears), len(umonths), row, col]) * np.nan
                # r_all = np.zeros([out_channels, len(uyears), len(umonths), row, col]) * np.nan

                for i, y in tqdm(enumerate(uyears)):
                    for j, m in enumerate(umonths): 

                        mdf.loc[id_start, "year"] = y
                        mdf.loc[id_start, "month"] = m
        
                        idx = (months == m) & (years == y)
                        val_days = days[idx]
                        data = data0[idx, :, :, :] #.astype(np.float32)
                        target = target0[idx, :, :, :] #.astype(np.float32)
                        # print(y, m, len(val_days))
        
                        data = torch.permute(data, (0, 3, 1, 2)) * (landmask == 0)
                        target = torch.permute(target, (0, 3, 1, 2)) * (landmask == 0)
                        # pred = target.clone() * 0
        
                        val_dataset = SeaiceDataset(data, target, val_days, 3, 1, exact = True)
                        val_loader = DataLoader(val_dataset, batch_size = 32)
                        n_sample = len(val_dataset)
        
                        for (inputs, target) in val_loader:
                            if device == "cuda":
                                inputs = inputs.cuda()        
                            outputs = model(inputs)
                            pred = outputs.cpu()

                        target = target.detach().numpy()
                        pred = pred.detach().numpy()
                        # print(target.shape, pred.shape)
                        
                        lm = np.array([landmask]).repeat(target.shape[0], axis = 0)    
        
                        sic_prd = pred[:, 2, :, :]*100
                        sic_obs = target[:, 2, :, :]*100
                        sic_max = np.nanmax(sic_obs, axis=0)
                        sic_th1 = -100 # observation threshold
                        sic_th2 = -100 # prediction threshold
        
                        vmax = [12, 12, 100, 24, 4]
                        vmin = [-12, -12, 0, 0, 0]
        
                        scaling = [30, 30, 100, 30, 8]
                        offset = [0, 0, 0, 0, 0]
                        cm = ['Spectral', 'Spectral', 'Blues', 'Reds', 'Blues']
                        titles = ["$U_{ice}$ (km/d)", "$V_{ice}$ (km/d)", "$SIC$ (%)", "$Velocity$ (km/day)", "$SIT$ (m)"]
        
                        df_region = []
        
                        for c in range(0, out_channels):
        
                            obs = ((target[:, c, :, :]) + offset[c]) *scaling[c]
                            prd = ((pred[:, c, :, :]) + offset[c]) *scaling[c] 
        
                            prd[(lm==0)==0] = np.nan
                            obs[(lm==0)==0] = np.nan
        
                            prd[:, sic_max <= 0] = np.nan
                            obs[:, sic_max <= 0] = np.nan
        
                            v_rmse = 0
                            v_r = 0
                            v_mbe = 0
                            v_mae = 0
                            v_skill = 0
                            
                            mdf.loc[id_start, f"RMSE{c}"] = RMSE(prd, obs) #v_rmse/n_samples #RMSE(prd, obs)
                            mdf.loc[id_start, f"R{c}"] = corr(prd, obs) #v_r/n_samples #corr(prd, obs
                            
                            rmse_all[c, i, j] = RMSE_grid(prd, obs)
                            # r_all[c, i, j] = corr_grid(prd, obs)   
        
                        id_start += 1
    
                rmse_total[keyname]= np.nanmean(rmse_all, axis=1) #RMSE_grid(prd_all, obs_all) #np.nanmean(rmse_all, axis=1)
                # r_total[keyname] = np.nanmean(r_all, axis=1) #corr_grid(prd_all, obs_all) #np.nanmean(r_all, axis=1)               
                rmse_all_mat[keyname] = rmse_all
                df[keyname] = mdf
    
                results_save = [rmse_total, rmse_all_mat, df, xx, yy]
                
                with open(result_path + f'/EMS_results_{year}_{keyname}.pkl', 'wb') as file:
                    pickle.dump(results_save, file)
        
    print("DONE! LET's MOVE ON TO THE NEXT STEP!")
    

if __name__ == '__main__':
    main()