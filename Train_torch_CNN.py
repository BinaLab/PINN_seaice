### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from tqdm import tqdm
import time
import pickle

import torch
    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_model import *

import argparse
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/',
        metavar='D',
        help='directory to download dataset to',
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='train_cnn_2019_2022_v2.pkl',
        help='filename of dataset',
    )    
    parser.add_argument(
        '--model-dir',
        default='../model',
        help='Model directory',
    )
    parser.add_argument(
        '--date',
        type=int,
        default=2019,
        help='year to exclude during the training process',
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    
    # Training settings
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        metavar='N',
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=16,
        help='input batch size for validation (default: 16)',
    )
    parser.add_argument(
        '--phy',
        type=str,
        default='nophy',
        help='filename of dataset',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='base learning rate (default: 0.01)',
    )    

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='local rank for distributed training',
    )

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

    
##########################################################################################

def main() -> None:
    
    ## Train parameters ##
    args = parse_args()
    
    data_path = args.data_dir
    data_file = args.data_file
    model_dir = args.model_dir
    date = args.date   

    n_epochs = args.epochs
    batch_size = args.batch_size  # size of each batch
    val_batch = args.val_batch_size  # size of validation batch size
    lr = args.base_lr

    phy = args.phy ## PHYSICS OR NOT

    with open(data_path + data_file, 'rb') as file:
        xx, yy, days, months, years, cnn_input, cnn_output = pickle.load(file)
    
    print("######## TRAINING DATA IS PREPARED (# of samples: {0}) ########".format(len(days)))

    net = Net()

    torch.cuda.empty_cache()
    
    if args.no_cuda:
        device = torch.device('cpu')
        device_name = 'cpu'
    else:
        device = torch.device('cuda')
        device_name = 'gpu'
        net = nn.DataParallel(net)        

    print(device)
    net.to(device)

    cnn_input = np.transpose(cnn_input, (0, 3, 1, 2))
    cnn_output = np.transpose(cnn_output, (0, 3, 1, 2))

    print(np.shape(cnn_input), np.shape(cnn_output))

    mask1 = (years == date) # Test samples
    mask2 = (days % 4 == 2) # Validation samples

    test_input = cnn_input[mask1, :, 41:, :-41]
    test_output = cnn_output[mask1, :, 41:, :-41]
    val_input = cnn_input[(~mask1)&(mask2), :, 41:, :-41]
    val_output = cnn_output[(~mask1)&(mask2), :, 41:, :-41]
    train_input = cnn_input[(~mask1)&(~mask2), :, 41:, :-41]
    train_output = cnn_output[(~mask1)&(~mask2), :, 41:, :-41]

    del mask1, mask2

    print(np.shape(train_input), np.shape(train_output), np.shape(val_input), np.shape(val_output), np.shape(test_input), np.shape(test_output))

    train_input = torch.tensor(train_input, dtype=torch.float32)
    train_output = torch.tensor(train_output, dtype=torch.float32)
    val_input = torch.tensor(val_input, dtype=torch.float32)
    val_output = torch.tensor(val_output, dtype=torch.float32)
    test_input = torch.tensor(test_input, dtype=torch.float32)
    test_output = torch.tensor(test_output, dtype=torch.float32)

    n_samples, row, col, in_channels = np.shape(train_input)
    _, _, _, out_channels = np.shape(train_output)    

    model_name = f"torch_cnn_lr{lr}_wo{date}_{phy}_{device_name}"

    if phy == "phy":
        loss_fn = physics_loss() # nn.L1Loss() #nn.CrossEntropyLoss()
    elif phy == "nophy":
        loss_fn = custom_loss() # nn.L1Loss() #nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    batch_start = torch.arange(0, len(train_input), batch_size)

    history = {'loss': [], 'val_loss': [], 'time': []}

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    
    t0 = time.time()
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        train_cnt = 0
        
        net.train()
        
        with tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = train_input[start:start+batch_size].to(device)
                y_batch = train_output[start:start+batch_size].to(device)
                # backward pass
                optimizer.zero_grad()
                # forward pass
                y_pred = net(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(loss=float(loss))
                train_loss += loss.item()
                train_cnt += 1
                
        net.eval()

        val_loss = 0.0
        val_cnt = 0
        
        for i, data in enumerate(np.arange(0, len(val_input), val_batch), 0):
            inputs = val_input[i:i+val_batch, :, :, :].to(device)
            outputs = net(inputs)
            truths = val_output[i:i+val_batch, :, :, :].to(device)
            val_loss += loss_fn(outputs, truths).item()
            val_cnt += 1

        history['loss'].append(train_loss/train_cnt)
        history['val_loss'].append(val_loss/val_cnt)
        history['time'].append(time.time() - t0)

    torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

    with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()