"""Pretty sure there's mistakes and redundancy in this code, but for learning,
it should serve as a starting point - without torchrun or torch.distributed.launch
or slurm to set the environment variables etc.

```bash
python omnixamples/distributed/a_basic/b_demo.py \
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
import os

import torch
import torch.distributed
import torch.multiprocessing as mp
from rich.pretty import pprint
from torch._C._distributed_c10d import ReduceOp

from omnivault.distributed.core import find_free_port, is_free_port
from omnivault.utils.reproducibility.seed import seed_all
from omnixamples.distributed.a_basic.a_setup import init_process
from omnixamples.distributed.a_basic.config import get_args_parser


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
    torch.distributed.all_reduce(data, op=ReduceOp.SUM, async_op=False)  # in-place
    logger.info(f"rank {dist_info_per_process.global_rank} data (after all-reduce): {data} with device {data.device}.")


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
