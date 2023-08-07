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
import torch.distributed as dist
from torch.utils import collect_env
 
from torch.utils.tensorboard import SummaryWriter

from torch_model import *

import argparse
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

try:
    from torch.cuda.amp import GradScaler

    TORCH_FP16 = True
except ImportError:
    TORCH_FP16 = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
) -> None:
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def parse_args() -> argparse.Namespace:
    """Get cmd line args."""
    
    # General settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='', #'D:\\PINN\\data\\',
        metavar='D',
        help='directory to download dataset to',
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='train_cnn_2019_2022_v3.pkl',
        help='filename of dataset',
    )    
    parser.add_argument(
        '--model-dir',
        default='../model',
        help='Model directory',
    )
    parser.add_argument(
        '--log-dir',
        default='./logs/torch_unet',
        help='TensorBoard/checkpoint directory',
    )
    parser.add_argument(
        '--date',
        type=int,
        default=2019,
        help='year to exclude during the training process',
    )
    parser.add_argument(
        '--checkpoint-format',
        default='checkpoint_unet_{epoch}.pth.tar',
        help='checkpoint file format',
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='epochs between checkpoints',
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )    
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        metavar='S',
        help='random seed (default: 42)',
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
        '--batches-per-allreduce',
        type=int,
        default=1,
        help='number of batches processed locally before '
        'executing allreduce across workers; it multiplies '
        'total batch size.',
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
        '--backend',
        type=str,
        default='nccl',
        help='backend for distribute training (default: nccl)',
    )
    # Set automatically by torch distributed launch
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
    
    """Main train and eval function."""
    args = parse_args()
    
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    torch.distributed.init_process_group(
        backend=args.backend,
        rank=WORLD_RANK, world_size=WORLD_SIZE        
    )

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    args.base_lr = (
        args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    )
    args.verbose = dist.get_rank() == 0

    if args.verbose:
        print('Collecting env info...')
        print(collect_env.get_pretty_env_info())
        print()

    for r in range(torch.distributed.get_world_size()):
        if r == torch.distributed.get_rank():
            print(
                f'Global rank {torch.distributed.get_rank()} initialized: '
                f'local_rank = {args.local_rank}, '
                f'world_size = {torch.distributed.get_world_size()}',
            )
        torch.distributed.barrier()


    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_format = os.path.join(args.log_dir, args.checkpoint_format)
    args.log_writer = SummaryWriter(args.log_dir) if args.verbose else None   
    
    
    
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
    
    
    #### READ DATA ##################################################################
    with open(data_path + data_file, 'rb') as file:
        xx, yy, days, months, years, cnn_input, cnn_output = pickle.load(file)

    print("######## TRAINING DATA IS PREPARED (# of samples: {0}) ########".format(len(days)))
    
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
    
    #############################################################################
    
    net = UNet()

    # torch.cuda.empty_cache()
    
    if args.no_cuda:
        device = torch.device('cpu')
        device_name = 'cpu'
    else:
        device = torch.device('cuda')
        device_name = 'gpu'
        # net = nn.DataParallel(net) 


    print(device)
    net.to(device)
    
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[args.local_rank],
    )

    model_name = f"torch_unet_lr{lr}_wo{date}_{phy}_{device_name}"

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
        
        # with tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        #     bar.set_description(f"Epoch {epoch}")
        for start in batch_start:
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
            # bar.set_postfix(loss=float(loss))
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
        
        if (epoch > 0 and dist.get_rank() == 0):
            if epoch % args.checkpoint_freq == 0:
                save_checkpoint(net.module, optimizer, args.checkpoint_format.format(epoch=epoch))
        
            history['loss'].append(train_loss/train_cnt)
            history['val_loss'].append(val_loss/val_cnt)
            history['time'].append(time.time() - t0)

            print(
                'Epoch {0} >>> Train loss {1:.3f}, Val loss: {2:.3f}, Training time: {3:.3f}'.format(
                    epoch, train_loss/train_cnt, val_loss/val_cnt, time.time()-t0
                ),
            )
    
    torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

    with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()
