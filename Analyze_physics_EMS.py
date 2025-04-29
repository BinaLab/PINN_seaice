### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
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

    print("#### DATA IS LOADED ####")

    # Unet
    if 'df' not in locals():
        df = {}
    if 'rmse_total' not in locals():
        rmse_total = {}
    if 'r_total' not in locals():
        r_total = {}   
    
    year = int(data_file[-11:-7])
    row, col = 256, 256
    
    for ratio in [0.2, 0.5, 1.0]:
        for phy_w in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:        
            for sat_w in [0, 0.2, 0.5, 1.0, 2.0, 5.0]: #, "lbunet", "ebunet", "unet", "cnn", "lg", "hycom"]:
    
                torch.cuda.empty_cache()
    
                in_channels, out_channels = 18, 3
                model = HIS_UNet(in_channels, out_channels, landmask, row, 3, phy)
                # res = f"{result_dir}\\test_torch_hisunet_satv7_all_2016_2022_r{ratio}_pw{pw}_{phy}_d3f1_gpu8_{year}.pkl"
                if year == 2015:
                    model_name = model_path + f"/torch_{model_type}_satv7_all_2016_2022_r{ratio}_pw{phy_w}_sw{sat_w}_d3f1_gpu4.pth"
                elif year == 2022:
                    model_name = model_path + f"/torch_{model_type}_satv7_all_2009_2015_r{ratio}_pw{phy_w}_sw{sat_w}_d3f1_gpu4.pth"
    
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
    
                mdf = pd.DataFrame({'year': np.ones(len(umonths))*year})
    
                rmse_all = np.zeros([out_channels, len(umonths), row, col]) * np.nan
                r_all = np.zeros([out_channels, len(umonths), row, col]) * np.nan
    
                for m in tqdm(umonths):    
    
                    idx = (months == m)
                    val_days = days[idx]
                    data = data0[idx, :, :, :] #.astype(np.float32)
                    target = target0[idx, :, :, :] #.astype(np.float32)
    
                    data = torch.permute(data, (0, 3, 1, 2)) * (landmask == 0)
                    target = torch.permute(target, (0, 3, 1, 2)) * (landmask == 0)
                    pred = target.clone() * 0
    
                    val_dataset = SeaiceDataset(data, target, val_days, 3, 1, exact = True)
                    val_loader = DataLoader(val_dataset, batch_size = 1)
    
                    for i, (inputs, _) in enumerate(val_loader):
                        if device == "cuda":
                            inputs = inputs.cuda()        
                        outputs = model(inputs)
                        pred[i] = outputs.cpu()
                    
                    lm = np.array([landmask]).repeat(target.shape[0], axis = 0)
    
                    id_start += pred.shape[0]     
    
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
               
                        mdf.loc[int(m-1), "Month"] = valmonths[0]
                        # mdf.loc[int(m-1), "MAE{0}".format(c)] = v_mae/n_samples #MAE(prd, obs)
                        mdf.loc[int(m-1), "RMSE{0}".format(c)] = RMSE(prd, obs) #v_rmse/n_samples #RMSE(prd, obs)
                        # mdf.loc[int(m-1), "MBE{0}".format(c)] =  MBE(prd, obs) # v_mbe/n_samples #MBE(prd, obs)
                        mdf.loc[int(m-1), "R{0}".format(c)] = corr(prd, obs) #v_r/n_samples #corr(prd, obs
                        # mdf.loc[int(m-1), "Skill{0}".format(c)] =v_skill/n_samples  #skill(prd, obs)
                        
                        rmse_all[c, m-1] = RMSE_grid(prd, obs)
                        r_all[c, m-1] = corr_grid(prd, obs)   
    
                    id_start += pred.shape[0]
    
                rmse_total[keyname]= np.nanmean(rmse_all, axis=1) #RMSE_grid(prd_all, obs_all) #np.nanmean(rmse_all, axis=1)
                r_total[keyname] = np.nanmean(r_all, axis=1) #corr_grid(prd_all, obs_all) #np.nanmean(r_all, axis=1)
    
                df[keyname] = mdf
    
                # mdf.to_csv(res.replace(".pkl", ".csv"))
    
    results_save = [rmse_total, r_total, df]
    
    with open(result_path + '/physics_EMS_results.pkl', 'wb') as file:
        pickle.dump(train_save, file)
        
    print("DONE! LET's MOVE ON TO THE NEXT STEP!")
    

if __name__ == '__main__':
    main()