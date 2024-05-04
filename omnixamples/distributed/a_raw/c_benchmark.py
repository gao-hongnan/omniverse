from __future__ import annotations

import argparse
import os
import timeit

import torch
import torch.distributed
import torch.multiprocessing as mp
from rich.pretty import pprint
from torch._C._distributed_c10d import ReduceOp

from omnivault.constants.dtype import TorchDtype
from omnivault.constants.memory import MemoryUnit
from omnivault.distributed.core import find_free_port, is_free_port
from omnivault.utils.reproducibility.seed import seed_all
from omnixamples.distributed.a_raw.a_setup import init_process
from omnixamples.distributed.a_raw.config import get_args_parser

VRAM_SIZES_IN_BYTES = {
    "512KB": 512 * MemoryUnit.KB,  # 512_000
    "1MB": 1 * MemoryUnit.MB,  # 1_000_000
    "10MB": 10 * MemoryUnit.MB,
    # "50MB": 50 * MemoryUnit.MB,
    # "100MB": 100 * MemoryUnit.MB,
    # "1GB": 1 * MemoryUnit.GB,
}

DTYPES = [torch.float32]  # 4bytes


def create_tensor_of_vram_size(dtype: torch.dtype, vram_size_in_bytes: int) -> torch.Tensor:
    """Create a tensor of the specified dtype that fits in the VRAM size."""

    # 1. For example, if `dtype` is `torch.float32` then `bytes_per_element` is 4.
    #    It returns the bytes needed to store a single element of the tensor.
    if dtype.is_floating_point:
        bytes_per_element = torch.finfo(dtype).bits // MemoryUnit.BYTE
        assert (
            bytes_per_element == torch.tensor([], dtype=dtype).element_size()
        )  # TODO: may be inefficient adding this assertion
    else:
        bytes_per_element = torch.iinfo(dtype).bits // MemoryUnit.BYTE

    # 2. Simple math, we need to find the number of elements required to
    #    "consume" the target vram. For example, if we want to consume 10MB of
    #    vram and each element is 4 bytes (float32), then we need ~2.5 million
    #    elements derived from 10MB / 4 bytes per element.
    total_elements_needed = int(vram_size_in_bytes / bytes_per_element)

    # 3. Create 1D tensor with the required number of elements.
    tensor = torch.empty(total_elements_needed, dtype=dtype)
    assert tensor.size() == (
        total_elements_needed,
    ), f"Expected tensor size {total_elements_needed} but got {tensor.size()}."

    return tensor


def run_benchmarks(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    seed_all(dist_info_per_process.global_rank, seed_torch=True, set_torch_deterministic=False)  # seed each process

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    for vram_size_name, vram_size_in_bytes in VRAM_SIZES_IN_BYTES.items():
        logger.info(f"Running benchmark for VRAM size: {vram_size_name} ({vram_size_in_bytes} bytes).")
        for dtype in DTYPES:
            tensor = create_tensor_of_vram_size(dtype=dtype, vram_size_in_bytes=vram_size_in_bytes).to(device)
            logger.info(f"Created tensor of dtype {dtype} with size {tensor.size()} and device {tensor.device}.")
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # blocks host cpu calls until all CUDA kernels have completed execution, here we just want to wait for the tensor creation to complete on ALL ranks
            torch.distributed.barrier()

            start_time = timeit.default_timer()
            torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, async_op=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.distributed.barrier()

            end_time = timeit.default_timer()

            time_on_this_rank = end_time - start_time

            # torch.distributed.all_gather_object(time_on_this_rank, time_on_this_rank)



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
