#!/usr/bin/env sh

# Get master address and port
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

echo "${master_addr}:${master_port}" > master_info.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=2
export NPROC_PER_NODE=2
export NODE_RANK=0
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