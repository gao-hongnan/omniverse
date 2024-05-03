#!/usr/bin/env sh

# Read master address and port from the shared file
master_info=$(cat master_info.txt)
export MASTER_ADDR=$(echo $master_info | cut -d':' -f1)
export MASTER_PORT=$(echo $master_info | cut -d':' -f2)

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=2
export NPROC_PER_NODE=2
export NODE_RANK=1
export WORLD_SIZE=2

python omnixamples/distributed/01_raw/01_demo.py \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --world_size=$WORLD_SIZE \
    --backend=gloo \
    --init_method="env://"