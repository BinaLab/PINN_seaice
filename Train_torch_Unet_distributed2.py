### PREDICT ONLY SEA ICE U & V

# Ignore warning
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import math

from tqdm import tqdm
import time
import pickle

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils import collect_env
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
 
from torch.utils.tensorboard import SummaryWriter

from torch_model import *

import argparse
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# try:
#     from torch.cuda.amp import GradScaler

#     TORCH_FP16 = True
# except ImportError:
#     TORCH_FP16 = False


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
        default='../data/', #'D:\\PINN\\data\\',
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
    # parser.add_argument(
    #     '--local_rank',
    #     type=int,
    #     default=0,
    #     help='local rank for distributed training',
    # )
    
    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def make_sampler_and_loader(args, train_dataset, val_dataset):
    """Create sampler and dataloader for train and val datasets."""
    torch.set_num_threads(4)
    kwargs: dict[str, Any] = (
        {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    )
    kwargs['prefetch_factor'] = 8
    kwargs['persistent_workers'] = True

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        **kwargs,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        **kwargs,
    )

    return train_sampler, train_loader, val_sampler, val_loader

# def init_processes(backend):
#     dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
#     run(backend)
    
class Metric:
    """Metric tracking class."""

    def __init__(self, name: str):
        """Init Metric."""
        self.name = name
        self.total = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        """Update metric.

        Args:
            val (float): new value to add.
            n (int): weight of new value.
        """
        dist.all_reduce(val, async_op=False)
        self.total += val.cpu() / dist.get_world_size()
        self.n += n

    @property
    def avg(self) -> torch.Tensor:
        """Get average of metric."""
        return self.total / self.n
    
def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_sampler: torch.utils.data.distributed.DistributedSampler,
    args,
):
    
    """Train model."""
    model.train()
    train_sampler.set_epoch(epoch)
    
    mini_step = 0
    step_loss = torch.tensor(0.0).to('cuda' if args.cuda else 'cpu')
    train_loss = Metric('train_loss')
    t0 = time.time()
    
    with tqdm(
        total=math.ceil(len(train_loader) / args.batches_per_allreduce),
        bar_format='{l_bar}{bar:10}{r_bar}',
        desc=f'Epoch {epoch:3d}/{args.epochs:3d}',
        disable=not args.verbose,
    ) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            mini_step += 1
            if args.cuda:
                data, target = data.cuda(), target.cuda()
                
            output = model(data)
            loss = loss_func(output, target)

            with torch.no_grad():
                step_loss += loss

            loss = loss / args.batches_per_allreduce

            if (
                mini_step % args.batches_per_allreduce == 0
                or batch_idx + 1 == len(train_loader)
            ):
                loss.backward()
            else:
                with model.no_sync():  # type: ignore
                    loss.backward()

            if (
                mini_step % args.batches_per_allreduce == 0
                or batch_idx + 1 == len(train_loader)
            ):

                optimizer.step()
                optimizer.zero_grad()
                
                train_loss.update(step_loss / mini_step)
                step_loss.zero_()

                t.set_postfix_str('loss: {:.4f}'.format(train_loss.avg))
                t.update(1)
                mini_step = 0

    if args.log_writer is not None:
        args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        
    return train_loss.avg


def test(
    epoch: int,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    args
):
    """Test the model."""
    model.eval()
    val_loss = Metric('val_loss')

    with tqdm(
        total=len(val_loader),
        bar_format='{l_bar}{bar:10}|{postfix}',
        desc='             ',
        disable=not args.verbose
    ) as t:
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(loss_func(output, target))

                t.update(1)
                if i + 1 == len(val_loader):
                    t.set_postfix_str(
                        'val_loss: {:.4f}'.format(val_loss.avg),
                        refresh=False,
                    )

    if args.log_writer is not None:
        args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
    return val_loss.avg
    
##########################################################################################

def main() -> None:    
    
    """Main train and eval function."""
    args = parse_args()

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method='env://',
    )

    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    # args.base_lr = (
    #     args.base_lr * dist.get_world_size() * args.batches_per_allreduce
    # )
    
    args.verbose = dist.get_rank() == 0
    world_size = int(os.environ['WORLD_SIZE'])

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

    val_input = cnn_input[(~mask1)&(mask2), :, 41:, :-41]
    val_output = cnn_output[(~mask1)&(mask2), :, 41:, :-41]
    train_input = cnn_input[(~mask1)&(~mask2), :, 41:, :-41]
    train_output = cnn_output[(~mask1)&(~mask2), :, 41:, :-41]
    # test_input = cnn_input[mask1, :, 41:, :-41]
    # test_output = cnn_output[mask1, :, 41:, :-41]
    
    print(np.shape(train_input), np.shape(train_output), np.shape(val_input), np.shape(val_output))
    
    train_input = torch.tensor(train_input, dtype=torch.float32)
    train_output = torch.tensor(train_output, dtype=torch.float32)
    val_input = torch.tensor(val_input, dtype=torch.float32)
    val_output = torch.tensor(val_output, dtype=torch.float32)
    # test_input = torch.tensor(test_input, dtype=torch.float32)
    # test_output = torch.tensor(test_output, dtype=torch.float32)   
    
    train_dataset = TensorDataset(train_input, train_output)
    val_dataset = TensorDataset(val_input, val_output)
    
    train_sampler, train_loader, _, val_loader = make_sampler_and_loader(args, train_dataset, val_dataset)

    n_samples, row, col, in_channels = train_input.size()
    _, _, _, out_channels = train_output.size()
    
    del train_input, train_output, val_input, val_output, mask1, mask2
    
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

    model_name = f"torch_unet_lr{lr}_wo{date}_{phy}_{device_name}{world_size}"

    if phy == "phy":
        loss_fn = physics_loss() # nn.L1Loss() #nn.CrossEntropyLoss()
    elif phy == "nophy":
        loss_fn = custom_loss() # nn.L1Loss() #nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    history = {'loss': [], 'val_loss': [], 'time': []}

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    
    t0 = time.time()
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        train_cnt = 0
        
        net.train()
        
        train_loss = train(
            epoch,
            net,
            optimizer,
            loss_fn,
            train_loader,
            train_sampler,
            args,
        )
        
        val_loss = test(epoch, net, loss_fn, val_loader, args)
        
        if (epoch > 0 and dist.get_rank() == 0):
            if epoch % args.checkpoint_freq == 0:
                save_checkpoint(net.module, optimizer, args.checkpoint_format.format(epoch=epoch))
        
            history['loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['time'].append(time.time() - t0)
    
    torch.save(net.state_dict(), f'{model_dir}/{model_name}.pth')

    with open(f'{model_dir}/history_{model_name}.pkl', 'wb') as file:
        pickle.dump(history, file)

if __name__ == '__main__':
    main()
