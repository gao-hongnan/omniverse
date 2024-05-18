"""Setup for distributed training. Init process without torchrun or slurm info.

In DDP just imagine all your function is replicated across all processes.
1. init_process(1)
2. init_process(2)
3. init_process(3)
4. init_process(4)
```

For safety net, sync all processes before proceeding - for example in
`configure_logger` there is an create directory operation which maybe should be
done by only master rank. Nevertheless, consider the fact that you don't
sync barrier, then you might run into problem of another rank process wanting
to write to the same directory before it is created by master rank.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import torch
import torch.distributed

from omnivault.distributed.core import get_hostname, get_process_id
from omnivault.distributed.dist_info import DistInfoPerProcess
from omnivault.distributed.logger import configure_logger


def init_process(
    local_rank: int, args: argparse.Namespace, logger: logging.Logger | None = None
) -> Tuple[logging.Logger, DistInfoPerProcess]:
    """This rank should be the rank of the process spawned by `mp.spawn`."""
    # NOTE: knowing `nnodes`, `node_rank` and `nproc_per_node` is sufficient to derive most of other env.
    nnodes = args.nnodes
    nproc_per_node = args.nproc_per_node  # local_world_size

    # NOTE: all these validations can be done via pydantic but for simplicity we do it here.
    if torch.cuda.is_available():
        assert nproc_per_node == torch.cuda.device_count()
    node_rank = args.node_rank

    world_size = args.world_size
    assert world_size == nnodes * nproc_per_node

    global_rank = local_rank + node_rank * nproc_per_node
    assert local_rank == global_rank % nproc_per_node

    hostname = get_hostname()
    process_id = get_process_id()

    dist_info_per_process = DistInfoPerProcess(
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        node_rank=node_rank,
        world_size=world_size,
        backend=args.backend,
        init_method=args.init_method,
        global_rank=global_rank,
        local_rank=local_rank,
        local_world_size=nproc_per_node,
        hostname=hostname,
        process_id=process_id,
    )

    if logger is None:
        # NOTE: this is global rank configuration for logger
        logger = configure_logger(
            rank=dist_info_per_process.global_rank, log_dir=args.log_dir, log_on_master_or_all=args.log_on_master_or_all
        )

    torch.distributed.init_process_group(
        backend=args.backend,
        rank=dist_info_per_process.global_rank,
        world_size=dist_info_per_process.world_size,
        init_method=args.init_method,
    )

    torch.distributed.barrier()
    # NOTE: set device should be for local rank, not global rank, else you run
    # into ordinal out of device error.
    torch.cuda.set_device(dist_info_per_process.local_rank) if torch.cuda.is_available() else None
    return logger, dist_info_per_process
