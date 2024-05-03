"""Pretty sure there's mistakes and redundancy in this code, but for learning,
it should serve as a starting point - without torchrun or torch.distributed.launch
or slurm to set the environment variables etc.

```bash
python omnixamples/distributed/01_raw/01_demo.py \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --world_size=4 \
    --backend=gloo \
    --init_method="env://"
```
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
import torch.multiprocessing as mp
from rich.logging import RichHandler
from rich.pretty import pprint
from torch._C._distributed_c10d import ReduceOp

from omnivault.distributed.core import find_free_port, get_hostname, get_process_id, is_free_port
from omnivault.distributed.dist_info import DistInfoPerProcess
from omnivault.distributed.logger import configure_logger
from omnivault.utils.reproducibility.seed import seed_all


def configure_logger_all_gather() -> None:
    ...


# NOTE: In DDP just imagine all your function is replicated across all processes.
# 1. init_process(1)
# 2. init_process(2)
# 3. init_process(3)
# 4. init_process(4) ...
def init_process(
    local_rank: int, args: argparse.Namespace, logger: logging.Logger | None = None
) -> Tuple[logging.Logger, DistInfoPerProcess]:
    """This rank should be the rank of the process spawned by `mp.spawn`."""
    # NOTE: knowing `nnodes`, `node_rank` and `nproc_per_node` is sufficient to derive most of other env.
    nnodes = args.nnodes
    nproc_per_node = args.nproc_per_node  # local_world_size
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

    # NOTE: safety net, sync all processes before proceeding - for example in
    # `configure_logger` there is an create directory operation which maybe should be
    # done by only master rank. Nevertheless, consider the fact that you don't
    # sync barrier, then you might run into problem of another rank process wanting
    # to write to the same directory before it is created by master rank.
    torch.distributed.barrier()
    # NOTE: set device should be for local rank, not global rank, else you run
    # into ordinal out of device error.
    torch.cuda.set_device(dist_info_per_process.local_rank) if torch.cuda.is_available() else None
    return logger, dist_info_per_process


def run(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    seed_all(dist_info_per_process.global_rank, seed_torch=True, set_torch_deterministic=False)  # seed each process

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    data = torch.randint(low=0, high=10, size=(3,)).to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.distributed.barrier()
    logger.info(f"rank {dist_info_per_process.global_rank} data (before all-reduce): {data} with device {data.device}.")
    torch.distributed.all_reduce(data, op=ReduceOp.SUM, async_op=False)
    logger.info(f"rank {dist_info_per_process.global_rank} data (after all-reduce): {data} with device {data.device}.")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed Training Demo")

    # LOGGING
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store logs.")
    parser.add_argument("--log_level", type=int, default=logging.INFO, help="Logging level.")
    parser.add_argument(
        "--log_on_master_or_all",
        action="store_true",
        help="Whether to log only on master rank or all ranks.",
    )

    # DISTRIBUTED
    # 1. Mandatory Environment Variables, note if use `torchrun` then not all are mandatory.
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address.")
    parser.add_argument("--master_port", type=str, default="29500", help="Master port.")
    parser.add_argument("--nnodes", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--nproc_per_node", type=int, required=True, help="Number of processes per node.")
    parser.add_argument("--node_rank", type=int, required=True, help="Node rank.")

    # 2. Optional Environment Variables, can be derived from above.
    parser.add_argument("--world_size", type=int, required=True, help="Total number of processes.")

    # 3. Initialization of Process Group
    parser.add_argument("--backend", type=str, default="gloo", help="Backend for distributed training.")
    parser.add_argument(
        "--init_method", type=str, default="env://", help="Initialization method for distributed training."
    )

    return parser


if __name__ == "__main__":
    # MANUAL/RAW NO TORCHRUN OR SLURM OR TORCH DISTRIBUTED LAUNCHER
    # torchrun --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 sandbox.py

    # NOTE: if you use torchrun then a lot of env variables are auto
    # set when you pass in the command line arguments to torchrun.

    args = get_args_parser().parse_args()
    pprint(args)

    master_addr, master_port = args.master_addr, args.master_port
    if not is_free_port(int(master_port)):
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    mp.spawn(
        fn=run,
        args=(args,),
        nprocs=args.nproc_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )  # type: ignore[no-untyped-call]
