"""
python omnixamples/distributed/a_raw/ablations.py \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --world_size=4 \
    --backend=gloo \
    --init_method="env://" \
    --run_with_no_barrier
"""

from __future__ import annotations

import argparse
import os
import torch
import torch.multiprocessing as mp
from rich.pretty import pprint

from omnivault.distributed.core import find_free_port, is_free_port
from omnixamples.distributed.a_raw.a_setup import init_process
from omnixamples.distributed.a_raw.config import get_args_parser


def run_with_no_barrier(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    results = []

    logger.info("I HAVE NO BARRIER DUDE!")

    # NOTE: add `torch.distributed.barrier()` here if you want to synchronize all processes
    results.append([1, 2, 3])

    logger.info(f"Results: {results}")


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--run_with_no_barrier", action="store_true", help="Run with no barrier.")
    parser.add_argument("--barrier", action="store_true", help="Run with barrier.")

    args = parser.parse_args()
    pprint(args)

    master_addr, master_port = args.master_addr, args.master_port
    if not is_free_port(int(master_port)):
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    if args.run_with_no_barrier:
        mp.spawn(
            fn=run_with_no_barrier,
            args=(args,),
            nprocs=args.nproc_per_node,
            join=True,
            daemon=False,
            start_method="spawn",
        )  # type: ignore[no-untyped-call]
