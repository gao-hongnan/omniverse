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

from omnivault.distributed.core import (
    find_free_port,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    is_free_port,
)
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
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    logger: logging.Logger | None = None,
) -> Tuple[logging.Logger, DistInfoPerProcess]:
    """Wrapper for `init_process_group`."""
    # NOTE: this os env can be modularized, but we show here for simplicity.

    if logger is None:
        logger = configure_logger(rank=rank, log_dir="logs", log_on_master_or_all=False)

    torch.distributed.init_process_group(
        backend=args.backend, rank=rank, world_size=world_size, init_method=args.init_method
    )
    #torch.distributed.barrier()
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    pprint(torch.distributed.get_rank())

    logger.info(f"Rank {rank}\n" f"World_size {world_size}\n" f"Backend: {torch.distributed.get_backend()}")

    local_world_size = get_local_world_size()
    local_rank = get_local_rank()
    global_rank = get_global_rank()

    assert rank == global_rank

    dist_info_per_process = DistInfoPerProcess(
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        node_rank=args.node_rank,
        world_size=args.world_size,
        backend=args.backend,
        init_method=args.init_method,
        global_rank=global_rank,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )
    return logger, dist_info_per_process


def distributed_demo(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """This rank should be the rank of the process spawned by `mp.spawn`."""
    logger, dist_info_per_process = init_process(rank, world_size, args=args)

    pprint(dist_info_per_process)

    seed_all(rank, seed_torch=True, set_torch_deterministic=False)  # seed each process

    data = torch.randint(low=0, high=10, size=(3,)).to(rank)
    torch.distributed.barrier()
    logger.info(f"rank {rank} data (before all-reduce): {data} with device {data.device}.")
    torch.distributed.all_reduce(data, op=ReduceOp.SUM, async_op=False)
    torch.distributed.barrier()
    logger.info(f"rank {rank} data (after all-reduce): {data} with device {data.device}.")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed Training Demo")

    # LOGGING
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store logs.")
    parser.add_argument("--log_level", type=int, default=logging.INFO, help="Logging level.")
    parser.add_argument(
        "--log_on_master_or_all",
        type=bool,
        default=True,
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
    # python --log_on_master_or_all=False --master_addr=localhost --master_port=29500 --world_size=4 sandbox.py

    args = get_args_parser().parse_args()
    pprint(args)

    nnodes = args.nnodes
    nproc_per_node = args.nproc_per_node
    node_rank = args.node_rank
    world_size = args.world_size

    # NOTE: if you use torchrun then a lot of env variables are auto
    # set when you pass in the command line arguments to torchrun.

    master_addr, master_port = args.master_addr, args.master_port
    # if not is_free_port(int(master_port)):
    #     master_port = find_free_port()

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    mp.spawn(
        fn=distributed_demo,
        args=(world_size, args),
        nprocs=nproc_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )  # type: ignore[no-untyped-call]

    # processes = []
    # mp.set_start_method("spawn")

    # logger = configure_logger("main")

    # for rank in range(world_size):
    #     p = mp.Process(
    #         target=init_process,
    #         args=(rank, world_size, "nccl", logger,)
    #         kwargs={"init_method": "env://"},
    #     )
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
