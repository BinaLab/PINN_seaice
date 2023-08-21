#!/bin/bash
HOST=$1
NODES=$2
LOCAL_RANK=${PMI_RANK}
torchrun --nproc_per_node=4 --nnodes=$NODES --node_rank=${LOCAL_RANK} --master_addr=$HOST Train_torch_CNN_distributed2.py
