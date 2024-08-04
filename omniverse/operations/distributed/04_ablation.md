# Ablations

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/blob/88e2c1743a4ea01c1756eb3fa44639f98d77ac83/omnixamples/distributed/a_basic/d_ablation.py)

```{contents}
:local:
```

## Barrier

```python
from __future__ import annotations

import argparse
import os

import torch
import torch.multiprocessing as mp
from rich.pretty import pprint

from omnivault.distributed.core import find_free_port, is_free_port
from omnixamples.distributed.a_basic.a_setup import init_process
from omnixamples.distributed.a_basic.config import get_args_parser


def run_with_no_barrier(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    results = []

    logger.info("I HAVE NO BARRIER DUDE!")

    # NOTE: add `torch.distributed.barrier()` here if you want to synchronize all processes
    results.append([1, 2, 3])

    logger.info(f"Results: {results}")


def run_with_barrier(local_rank: int, args: argparse.Namespace) -> None:
    logger, dist_info_per_process = init_process(local_rank, args=args)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")

    results = []

    # We use barrier to synchronize all processes before computation.
    # A barrier acts as a checkpoint in the code. When a process reaches this
    # checkpoint, it must wait until all other processes in the group also reach this
    # checkpoint.
    logger.info("I HAVE BARRIER DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE...")
    torch.distributed.barrier()

    results.append([1, 2, 3])

    logger.info(f"Results: {results}")


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--run_with_no_barrier", action="store_true", help="Run with no barrier.")

    args = parser.parse_args()
    pprint(args)

    master_addr, master_port = args.master_addr, args.master_port
    if not is_free_port(int(master_port)):
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    target_fn = run_with_no_barrier if args.run_with_no_barrier else run_with_barrier

    mp.spawn(
        fn=target_fn,
        args=(args,),
        nprocs=args.nproc_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )  # type: ignore[no-untyped-call]
```

### No Distributed Barrier

If you run:

```bash
python omnixamples/distributed/a_basic/d_ablations.py \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --world_size=4 \
    --backend=gloo \
    --init_method="env://" \
    --run_with_no_barrier
```

Which is invokes `run_with_no_barrier`, we would sometimes see the below:

```python
2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                       ablations.py:32
2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                       ablations.py:32
2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                          ablations.py:36
2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                       ablations.py:32
2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                          ablations.py:36
2024-05-05 13:29:55 [INFO]: {
    "master_addr": "localhost",
    "master_port": "29500",
    "nnodes": 1,
    "nproc_per_node": 4,
    "node_rank": 0,
    "world_size": 4,
    "backend": "gloo",
    "init_method": "env://",
    "global_rank": 3,
    "local_world_size": 4,
    "local_rank": 3,
    "hostname": "Hongnans-Mac-mini.local",
    "process_id": 20647
}                                                                                         ablations.py:28
2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                          ablations.py:36
2024-05-05 13:29:55 [INFO]: I HAVE NO BARRIER DUDE!                                       ablations.py:32
2024-05-05 13:29:55 [INFO]: Results: [[1, 2, 3]]                                          ablations.py:36
```

You see that even when printing the `results` is after the
`I HAVE NO BARRIER DUDE!` message, the results are printed before the message.
This is because there is no barrier to synchronize the processes. This does not
always happen since in distributed systems, since the underlying distributed
system is asynchronous/concurrent in nature and the order of execution is not
guaranteed. Just think of each process being an independent entity and is
governed by say, the underlying resources (i.e. CPU, memory, etc.) and hence
they may not start at the exact same time. Consequently, this eliminates any
_race conditions_ that may arise where you know, 1 process happens to be faster
than the other.

To resolve this, we can add a `torch.distributed.barrier()` to synchronize the
processes before printing the results. Honestly I do not know enough about it to
discuss on a rigorous level, but I think you can have a mental model like below:

1. We do a **_point of synchronization_** at the `torch.distributed.barrier()`.
   This means when each process reaches this checkpoint, it must wait until all
   other processes in the group also reach this checkpoint.

    This means there is a _waiting_ period for all processes to reach the
    checkpoint before proceeding.

    ```python
    logger.info("I HAVE BARRIER DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE...")
    torch.distributed.barrier()
    ```

    So with this barrier, we would guarantee that the message _I HAVE BARRIER
    DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE..._ is printed before the
    results because every process must reach the barrier before proceeding.

2. Once all processes reach the barrier, they can all _simultaneously released_
   to continue to the next block of code.

    ```python
    results.append([1, 2, 3])
    logger.info(f"Results: {results}")
    ```

    Now all our processes will print this message after the barrier message.

### With Distributed Barrier

If you run:

```bash
python omnixamples/distributed/a_basic/d_ablations.py \
    --master_addr=localhost \
    --master_port=29500 \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --world_size=4 \
    --backend=gloo \
    --init_method="env://"
```

You would then see, the order of the messages are guaranteed:

```python
INFO     2024-08-04 16:45:34 [INFO]: I HAVE BARRIER DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE...                                                   d_ablations.py:65
INFO     2024-08-04 16:45:34 [INFO]: I HAVE BARRIER DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE...                                                   d_ablations.py:65
INFO     2024-08-04 16:45:34 [INFO]: I HAVE BARRIER DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE...                                                   d_ablations.py:65
INFO     2024-08-04 16:45:34 [INFO]: I HAVE BARRIER DUDE! WAITING FOR ALL PROCESSES TO SYNCHRONIZE...                                                   d_ablations.py:65
INFO     2024-08-04 16:45:34 [INFO]: Results: [[1, 2, 3]]                                                                                               d_ablations.py:70
INFO     2024-08-04 16:45:34 [INFO]: Results: [[1, 2, 3]]                                                                                               d_ablations.py:70
INFO     2024-08-04 16:45:34 [INFO]: Results: [[1, 2, 3]]                                                                                               d_ablations.py:70
INFO     2024-08-04 16:45:34 [INFO]: Results: [[1, 2, 3]]                                                                                               d_ablations.py:70
```
