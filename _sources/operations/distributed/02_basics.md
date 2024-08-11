# Basics Of Distributed Data Parallelism

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/blob/88e2c1743a4ea01c1756eb3fa44639f98d77ac83/omnixamples/distributed/a_basic/b_demo.py)

```{contents}
:local:
```

**DistributedDataParallel (DDP)** provides module-level data parallelism that's
scalable across multiple machines. For effective use, we quote pytorch's
documentation[^pytorch-distributed-data-parallel-tutorial]:

1. Launch multiple processes, initializing one DDP instance for each.
2. DDP leverages collective communications via the
   [torch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
   module for gradient and buffer synchronization.
3. An autograd hook is registered for each parameter determined by
   **`model.parameters()`**. This hook is activated when the gradient is
   computed during the backward pass, signaling DDP to synchronize gradients
   across processes. Further insights can be found in the
   [DDP design note](https://pytorch.org/docs/master/notes/ddp.html).

Traditionally if you have `train.py` script that trains a model on a single GPU
using say a `Trainer` class, now if you were to use DDP, you can think of it as
replicating the `Trainer` class across multiple processes and note that the
_model_ and _optimizer_ are
[replicated](https://engineering.fb.com/2021/07/15/open-source/fsdp/) across
processes as well. However, the data is not replicated across processes.
Instead, each process gets a subset of the data to work on. The gradients are
synchronized across processes using collective communications. Note that
replicating the model and optimizer across all ranks has overhead in gpu memory,
and if your model and optimizer are very large then you may need to consider
techniques like fsdp or combining DDP with model parallelism.

## Setting Up

In this post, we set up a simple example to demonstrate the use of DDP without
the use of the `torch.distributed.launch`, `torchrun` or SLURM.

We need to know a few things before we start:

-   The number of nodes in the cluster, $N$.
-   The number of processes per node - the number of processes that will run on
    each node. In other words, the number of GPUs per node, $G$. Please just
    keep it the same for all nodes.
-   The node rank, $n$, which is the index of the node in the cluster. The node
    rank is $0$-indexed.

Knowing these three things is sufficient for one to set up DDP, even in a
distributed setting. Everything else like world size, local rank, and global
rank can be derived from these three. Additionally, you would need the
`MASTER_ADDR` and `MASTER_PORT` environment variables. These are used to set up
the rendezvous point for the processes to communicate with each other - which is
a must in a multi-node setup.

### Demo Code

Note that if you are using say, 1 node, and 8 GPUs, but you only want to use 4
GPUs, then you can set `CUDA_VISIBLE_DEVICES` to `0,1,2,3` and then run the
script. The script will then only use the first 4 GPUs.

````{tab} **b_demo.py**
```python
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
```
````

````{tab} **config**
```python
import argparse
import logging


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
```
````

````{tab} **a_setup.py**
```python
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


# NOTE: In DDP just imagine all your function is replicated across all processes.
# 1. init_process(1)
# 2. init_process(2)
# 3. init_process(3)
# 4. init_process(4)
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
```
````

````{tab} **dist_info.py**
```python
from pydantic import BaseModel, Field
from rich.pretty import pprint


class DistInfoPerProcess(BaseModel):
    """Assumes one process is one worker/gpu. Immutable and should only
    perform data validation."""

    master_addr: str = Field(
        ...,
        description="""
                    This is `MASTER_ADDR` which refers to the IP address (or hostname)
                    of the machine or node where the rank 0 process is running.
                    It acts as the reference point for all other nodes and GPUs
                    in the distributed setup. All other processes will connect
                    to this address for synchronization and communication.
                    """,
    )
    master_port: str = Field(
        ...,
        description="Denotes an available port on the `MASTER_ADDR` machine. "
        "All processes will use this port number for communication.",
    )

    nnodes: int = Field(
        ...,
        description="Number of nodes in the distributed setup.",
    )

    nproc_per_node: int = Field(
        ...,
        description="Number of processes/gpus per node in the distributed setup.",
    )

    node_rank: int = Field(
        ...,
        description="Rank of the current node in the distributed setup.",
    )

    world_size: int = Field(
        ...,
        description="Total number of processes/gpus in the distributed setup.",
    )

    # 3. Initialization of Process Group
    backend: str = Field(
        ...,
        description="Backend for distributed training.",
    )

    init_method: str = Field(
        "env://",
        description="Initialization method for distributed training.",
    )

    # 4. Others
    global_rank: int = Field(
        ...,
        description="Rank of the current process/gpu in the distributed setup.",
    )
    local_world_size: int = Field(
        ...,
        description="Total number of processes/gpus in the local node.",
    )
    local_rank: int = Field(
        ...,
        description="Rank of the current process/gpu in the local node.",
    )

    # 5. System info
    hostname: str = Field(
        ...,
        description="Hostname of the current node.",
    )
    process_id: int = Field(
        ...,
        description="Process ID of the current process.",
    )

    def pretty_print(self) -> None:
        pprint(self)
```
````

### Command Line Arguments (CPU And Gloo Backend)

There's quite a fair bit of command line arguments, but most of them are simple
and will be understood shortly. To run the script you can do (for a single node)
and with CPU and backend `gloo`:

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

This will run the script on a single node with 4 processes. In the first tab
above, you can see the `torch.multiprocessing.spawn` function is used to spawn
the processes on the target function `run`, with `nprocs` defined as the number
of processes per node. The `run` function is where the actual work is done. A
side note is that the `torch.multiprocessing.spawn` function takes in implicitly
a `local_rank` argument which is defined as `i` in the source code below. What
this means is that the function `run` corresponds to the `fn` argument in the
`torch.multiprocessing.spawn` and since the underlying source code does
`args=(fn, i, args, error_queue)` then it means our `run` function should take
in an integer argument as the first argument - which is the `local_rank`.

```python
def start_processes(
    fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"
):
    mp = multiprocessing.get_context(start_method)
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    if start_method != "spawn":
        msg = (
            "This method only supports start_method=spawn (got: %s).\n"
            "To use a different start_method use:\n\t\t"
            " torch.multiprocessing.start_processes(...)" % start_method
        )
        warnings.warn(msg)
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
```

### Process Group Initialization

The idea of process group can be intuitive. Consider 1 single node with 4 gpus,
and you want to run 4 processes, one on each gpu. The process group is the
**collection** of these 4 processes. With the same logic, consider 2 nodes, with
2 gpus each, now you would run 2 processes on each node, and the process group
is the **collection** of these 4 processes. The word collection is vague, but if
you add the definition of all processes within a process group can
**_communicate_** with each other, then it makes sense. Consequently, the
process group represents multiple worker processes that coordinate and
communicate with each other via a shared
**_master_**[^cs336-spring2024-assignment2-systems].

#### Master Port and Master Address

How does the shared master get defined. It is defined by its IP address and
port, and we see in the first tab above, we need to specify the `MASTER_ADDR`
and `MASTER_PORT`. For example, if you have 2 compute nodes on SLURM, then the
`MASTER_ADDR` would be the IP address of the first node (rank 0) and the
`MASTER_PORT` would be a free port on that node. The second node would then
connect to the first node using the `MASTER_ADDR` and `MASTER_PORT`.

#### Backend

So how should we initialize the process group? The `init_process_group` function
will be the responsible function to initialize the process group. One needs to
specify the `backend`, `rank`, `world_size`, and `init_method`, amongs others.
But of course if your `init_method` is `env://` then you don't need to specify
the `rank` and `world_size` as they are derived from the environment variables.
But we still pass in here for clarity.

The backend is the communication backend to use and usually is `gloo` for cpu
and `nccl` for gpu because it will use the NVIDIA Collective Communication
Library (NCCL) for communication.

The code snippet below from the `a_setup.py` file shows how the process group is
initialized:

```python
torch.distributed.init_process_group(
    backend=args.backend,
    rank=dist_info_per_process.global_rank,
    world_size=dist_info_per_process.world_size,
    init_method=args.init_method,
)
```

#### Global Rank And World Size

Now remember that we are not using any convenience cluster like SLURM. So we
need to derive the `global_rank` and `world_size` from the `local_rank`,
`node_rank`, and `nproc_per_node` in order to pass it to the
`init_process_group` function.

The logic is simple, if you have 2 nodes, each with 4 gpus, then world size is
just `2 * 4 = 8`. The global rank is then derived from knowing the `node_rank`,
for example, if your node rank is 0, and since `local_rank` is given by the
`mp.spawn` function from `0-3`, then the global rank is just
`local_rank + node_rank * nproc_per_node`. Similarly, if your node rank is 1,
then the global rank is also `local_rank + node_rank * nproc_per_node`.

```python
nnodes = args.nnodes
nproc_per_node = args.nproc_per_node  # local_world_size

# NOTE: all these validations can be done via pydantic but for simplicity we do it here.
if torch.cuda.is_available():
    assert nproc_per_node == torch.cuda.device_count()
node_rank = args.node_rank

world_size = args.world_size
assert world_size == nnodes * nproc_per_node

global_rank = local_rank + node_rank * nproc_per_node
```

#### Set CUDA Device

When running multi-GPU jobs, make sure that different ranks use different GPUs.
This won't be needed for CPU-only jobs.

Two methods to do that, and I used both - I mean I use both for clarity and not
worried about the overhead.

First method when moving tensor to device, you need to move it to local rank and
not global rank. For example, you can do `tensor.to(f"cuda:{local_rank}")`
because the common `tensor.to("cuda")` will no longer work in multi-GPU setup.

Second method is to set the device using `torch.cuda.set_device(local_rank)`
after you init the process group. This way, when you do `tensor.to("cuda")` it
will automatically move it to the specified
device[^cs336-spring2024-assignment2-systems].

### Distributed Information

Let's first define a pydantic class that holds information about the distributed
setup. This class will be immutable and will only perform data validation. The
purpose of this class is after creation of distributed setup, we can inject this
object to other classes or functions to use the distributed information.

If you click on the tab named `dist_info.py` above, you will see the code - and
after running the above command, you will see 4 outputs, from each process,
holding the below info:

````{tab} Process 0
```python
DistInfoPerProcess(
    master_addr='localhost',
    master_port='29500',
    nnodes=1,
    nproc_per_node=4,
    node_rank=0,
    world_size=4,
    backend='gloo',
    init_method='env://',
    global_rank=0,
    local_world_size=4,
    local_rank=0,
    hostname='Hongnans-Mac-mini.local',
    process_id=72041
)
```
````

````{tab} Process 1
```python
DistInfoPerProcess(
    master_addr='localhost',
    master_port='29500',
    nnodes=1,
    nproc_per_node=4,
    node_rank=0,
    world_size=4,
    backend='gloo',
    init_method='env://',
    global_rank=1,
    local_world_size=4,
    local_rank=1,
    hostname='Hongnans-Mac-mini.local',
    process_id=72042
)
```
````

````{tab} Process 2
```python
DistInfoPerProcess(
    master_addr='localhost',
    master_port='29500',
    nnodes=1,
    nproc_per_node=4,
    node_rank=0,
    world_size=4,
    backend='gloo',
    init_method='env://',
    global_rank=2,
    local_world_size=4,
    local_rank=2,
    hostname='Hongnans-Mac-mini.local',
    process_id=72043
)
```
````

````{tab} Process 3
```python
DistInfoPerProcess(
    master_addr='localhost',
    master_port='29500',
    nnodes=1,
    nproc_per_node=4,
    node_rank=0,
    world_size=4,
    backend='gloo',
    init_method='env://',
    global_rank=3,
    local_world_size=4,
    local_rank=3,
    hostname='Hongnans-Mac-mini.local',
    process_id=72044
)
```
````

Of course it is difficult to log pydantic objects into log file, so you can just
`model_dump_json(indent=4)` to get a json string and log that.

## Multi-Node Setup With CUDA

Now consider the case where we have 2 nodes, and 1 GPU on each node. Let's see
if our setup works! Note that we use backend as `gloo` but in practice you would
use `nccl` for CUDA as it is faster (?) and includes more modes of
communication.

We first have two nodes, so we need two scripts - one for the master and one for
the worker. The master script will write the `MASTER_ADDR` and `MASTER_PORT` to
a file, and the worker script will read from this file. The master script will
start the process group and the worker script will join the process group.

````{tab} **01_demo_start_master.sh**
```bash
#!/usr/bin/env sh

# Get master address and port
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

echo "${MASTER_ADDR}:${MASTER_PORT}" > master_info.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=2
export NPROC_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=2

python omnixamples/distributed/a_basic/b_demo.py \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --world_size=$WORLD_SIZE \
    --backend=gloo \
    --init_method="env://"
```
````

````{tab} **01_demo_start_worker.sh**
```bash
#!/usr/bin/env sh

# Read master address and port from the shared file
master_info=$(cat master_info.txt)
export MASTER_ADDR=$(echo $master_info | cut -d':' -f1)
export MASTER_PORT=$(echo $master_info | cut -d':' -f2)

export PYTHONPATH=$PYTHONPATH:$(pwd)
export NNODES=2
export NPROC_PER_NODE=1
export NODE_RANK=1
export WORLD_SIZE=2

python omnixamples/distributed/a_basic/b_demo.py \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --world_size=$WORLD_SIZE \
    --backend=gloo \
    --init_method="env://"
```
````

Running them shows the output logs below for `process_0` and `process_1`:

````{tab} Process 0
```bash
Namespace(
    log_dir="logs",
    log_level=20,
    log_on_master_or_all=False,
    master_addr="10.0.23.218",
    master_port="1775",
    nnodes=2,
    nproc_per_node=1,
    node_rank=0,
    world_size=2,
    backend="gloo",
    init_method="env://",
)

2024-05-18 10:03:47 INFO     2024-05-18 10:03:47 [INFO]: {                                                           b_demo.py:37
                                 "master_addr": "10.0.23.218",
                                 "master_port": "1775",
                                 "nnodes": 2,
                                 "nproc_per_node": 1,
                                 "node_rank": 0,
                                 "world_size": 2,
                                 "backend": "gloo",
                                 "init_method": "env://",
                                 "global_rank": 0,
                                 "local_world_size": 1,
                                 "local_rank": 0,
                                 "hostname": "distributed-queue-st-g4dn2xlarge-1",
                                 "process_id": 4598
                             }

2024-05-18 10:03:48 INFO 2024-05-18 10:03:48 [INFO]: rank 0 data (before all-reduce): tensor([4, 9, 3], device='cuda:0') with device cuda:0.
b_demo.py:48
2024-05-18 10:03:48 INFO 2024-05-18 10:03:48 [INFO]: rank 0 data (after all-reduce): tensor([9, 18, 7], device='cuda:0') with device cuda:0.
b_demo.py:50
```
````

````{tab} Process 1

```bash
Namespace(
    log_dir="logs",
    log_level=20,
    log_on_master_or_all=False,
    master_addr="10.0.23.218",
    master_port="1775",
    nnodes=2,
    nproc_per_node=1,
    node_rank=1,
    world_size=2,
    backend="gloo",
    init_method="env://",
)
2024-05-18 10:03:47 INFO     2024-05-18 10:03:47 [INFO]:  b_demo.py:37
                             {
                                 "master_addr":
                             "10.0.23.218",
                                 "master_port": "1775",
                                 "nnodes": 2,
                                 "nproc_per_node": 1,
                                 "node_rank": 1,
                                 "world_size": 2,
                                 "backend": "gloo",
                                 "init_method": "env://",
                                 "global_rank": 1,
                                 "local_world_size": 1,
                                 "local_rank": 0,
                                 "hostname":
                             "distributed-queue-st-g4dn2x
                             large-2",
                                 "process_id": 4620
                             }
2024-05-18 10:03:48 INFO 2024-05-18 10:03:48 [INFO]: rank 1 data (before all-reduce): tensor([5, 9, 4], device='cuda:0') with device cuda:0.
b_demo.py:48
2024-05-18 10:03:48 INFO 2024-05-18 10:03:48 [INFO]: rank 1 data (after all-reduce): tensor([9, 18, 7], device='cuda:0') with device cuda:0.
b_demo.py:50
```
````

## References and Further Readings

-   [PyTorch: Distributed Data Parallel Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
-   [How Distributed Data Parallelism is Designed in PyTorch](https://pytorch.org/docs/stable/notes/ddp.html)
-   [CS336: Language Modeling from Scratch](https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/master/cs336_spring2024_assignment2_systems.pdf)

[^pytorch-distributed-data-parallel-tutorial]:
    [PyTorch: Distributed Data Parallel Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

[^cs336-spring2024-assignment2-systems]:
    [CS336: Language Modeling from Scratch](https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/master/cs336_spring2024_assignment2_systems.pdf)
