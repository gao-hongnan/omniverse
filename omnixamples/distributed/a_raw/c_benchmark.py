"""Note that for even more robust benchmarks, one can define a configurable
parameter `num_trials` and run the benchmark multiple times to get a
better estimate of the statitisics.

Note
----
1.   Can use torch's profiler for better cuda event measuring and better accuracy.
2.   If want to benchmark real matrices or tensors, then write function to generate
     tensor vram size based on tensor dimensions. If tensor is MxN = 100 x 200, then
     tensor size = M * N * bytes_per_element.

```bash
python omnixamples/distributed/a_raw/c_benchmark.py \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --world_size=4 \
    --backend=gloo \
    --init_method="env://" \
    --num_trials=5 \
    --warmup_trials=1
```
"""

from __future__ import annotations

import argparse
import copy
import os
import timeit
from typing import List, Tuple

import torch
import torch.distributed
import torch.multiprocessing as mp
from rich.pretty import pprint
from torch._C._distributed_c10d import ReduceOp

from omnivault.benchmark.create_tensor_of_size import create_tensor_of_vram_size
from omnivault.benchmark.statistics import calculate_statistics
from omnivault.constants.memory import MemoryUnit
from omnivault.distributed.core import find_free_port, is_free_port, is_master_rank, synchronize_and_barrier
from omnivault.utils.reproducibility.seed import seed_all
from omnixamples.distributed.a_raw.a_setup import init_process
from omnixamples.distributed.a_raw.config import get_args_parser

VRAM_SIZES_IN_BYTES = {
    "512KB": 512 * MemoryUnit.KB,  # 512_000
    "1MB": 1 * MemoryUnit.MB,  # 1_000_000
    # "10MB": 10 * MemoryUnit.MB,
    # "50MB": 50 * MemoryUnit.MB,
    # "100MB": 100 * MemoryUnit.MB,
    # "1GB": 1 * MemoryUnit.GB,
}

DTYPES = [  # note that float32 is 4 bytes per element so a 1D fp32 tensor with 10 elements will consume 40 bytes
    torch.float32
]


def time_all_reduce(tensor: torch.Tensor, world_size: int) -> Tuple[float, List[float]]:
    synchronize_and_barrier()  # blocks host cpu calls until all CUDA kernels have completed execution, here we just want to wait for the tensor creation to complete on ALL ranks

    start_time = timeit.default_timer()
    torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, async_op=False)
    synchronize_and_barrier()
    end_time = timeit.default_timer()

    time_on_this_rank: float = end_time - start_time  # time taken for all-reduce on this rank

    time_on_all_ranks = [0.0] * world_size
    torch.distributed.all_gather_object(object_list=time_on_all_ranks, obj=time_on_this_rank)
    return time_on_this_rank, time_on_all_ranks


def run_benchmarks(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    seed_all(dist_info_per_process.global_rank, seed_torch=True, set_torch_deterministic=False)  # seed each process

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    trials_results = []

    for vram_size_name, vram_size_in_bytes in VRAM_SIZES_IN_BYTES.items():
        for dtype in DTYPES:
            tensor = create_tensor_of_vram_size(dtype=dtype, vram_size_in_bytes=vram_size_in_bytes).to(device)
            if is_master_rank():
                logger.info(f"Running benchmark for VRAM size: {vram_size_name} ({vram_size_in_bytes} bytes).")
                logger.info(f"Created tensor of dtype {dtype} with size {tensor.size()} and device {tensor.device}.")

            if args.warmup_trials > 0:
                for _ in range(args.warmup_trials):
                    time_all_reduce(tensor=copy.deepcopy(tensor), world_size=dist_info_per_process.world_size)

            # NOTE: The real benchmark starts after the warmup trials.
            time_on_this_rank, time_on_all_ranks = time_all_reduce(
                tensor=tensor, world_size=dist_info_per_process.world_size
            )

            average_time_per_rank = sum(time_on_all_ranks) / dist_info_per_process.world_size

            statistics = calculate_statistics(data=time_on_all_ranks)
            if is_master_rank():  # only master rank will print this cause all ranks will have the same list
                logger.info(f"Time taken for all-reduce on all ranks: {time_on_all_ranks}.")
                logger.info(f"Average time taken for all-reduce on all ranks: {average_time_per_rank:.6f} seconds.")
                logger.info(f"Statistics for time taken for all-reduce on all ranks: {statistics}.\n")

            torch.distributed.barrier()
            # you want append results across all ranks
            trials_results.append(
                {
                    "global_rank": dist_info_per_process.global_rank,
                    "vram_size_name": vram_size_name,
                    "vram_size_in_bytes": vram_size_in_bytes,
                    "dtype": dtype,
                    "time_on_this_rank": time_on_this_rank,
                    "time_on_all_ranks": time_on_all_ranks,
                    "average_time_per_rank": average_time_per_rank,
                    "aggregated_statistics_across_all_ranks": statistics,
                }
            )
            pprint(trials_results)


if __name__ == "__main__":
    # MANUAL/RAW NO TORCHRUN OR SLURM OR TORCH DISTRIBUTED LAUNCHER
    # torchrun --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 sandbox.py

    # NOTE: if you use torchrun then a lot of env variables are auto
    # set when you pass in the command line arguments to torchrun.

    parser = get_args_parser()
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run the benchmark.")
    parser.add_argument("--warmup_trials", type=int, default=0, help="Number of warmup trials to run the benchmark.")

    args = parser.parse_args()
    pprint(args)

    master_addr, master_port = args.master_addr, args.master_port
    if not is_free_port(int(master_port)):
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    mp.spawn(
        fn=run_benchmarks,
        args=(args,),
        nprocs=args.nproc_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )  # type: ignore[no-untyped-call]
