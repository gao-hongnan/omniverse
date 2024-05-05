# A Simple Distributed Walkthrough

Assuming 2 nodes. One master and one worker - both compute nodes. Go into a
cluster say SLURM head node and ssh into both compute nodes to run the following
commands on each node.

```bash
bash omnixamples/distributed/01_raw/scripts/01_demo_start_master.sh
```

```bash
bash omnixamples/distributed/01_raw/scripts/01_demo_start_worker.sh
```

Try use slurm later on to compare much easier setup.

# WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH

## Setup

-   Single Node
-   4 GPUs
-   PBS

```python
"""
qsub -I -l select=1:ngpus=4 -P 11003281 -l walltime=24:00:00 -q ai
"""
import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich.logging import RichHandler
from multigpu import prepare_dataloader,load_train_objs, Trainer

def configure_logger(rank: int) -> logging.Logger:
    handlers = [logging.FileHandler(filename=f"process_{rank}.log")] # , RichHandler()]
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(f"Process-{rank}")


def init_env(**kwargs: Dict[str, Any]) -> None:
    """Initialize environment variables."""
    os.environ["MASTER_ADDR"] = kwargs.pop(
        "master_addr", "localhost"
    )  # IP address of the machine
    os.environ["MASTER_PORT"] = kwargs.pop("master_port", "12356")  # port number
    os.environ.update(kwargs)


def init_process(
    rank: int,
    world_size: int,
    backend: str,
    logger: logging.Logger,
    fn: Optional[Callable] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """Initialize the distributed environment via init_process_group."""

    logger = configure_logger(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **kwargs)
    logger.info(f"Initialized process group: Rank {rank} out of {world_size}")

    display_dist_info(rank, world_size, logger)

    if fn is not None:
        fn(rank, world_size)


def display_dist_info(rank: int, world_size: int, logger: logging.Logger) -> None:
    logger.info(f"Explicit Rank: {rank}")
    logger.info(f"Explicit World Size: {world_size}")
    logger.info(f"Machine Hostname: {socket.gethostname()}")
    logger.info(f"PyTorch Distributed Available: {dist.is_available()}")
    logger.info(f"World Size in Initialized Process Group: {dist.get_world_size()}")

    group_rank = dist.get_rank()
    logger.info(f"Rank within Default Process Group: {group_rank}")

def run(rank: int, world_size: int) -> None:
    """To be implemented."""
    ...


def main(world_size: int) -> None:
    """Main driver function."""
    init_env()
    processes = []
    mp.set_start_method("spawn")
    logger = configure_logger("main")

    for rank in range(world_size):
        p = mp.Process(
            target=init_process,
            args=(rank, world_size, "nccl", logger, run),
            kwargs={"init_method": "env://"},
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--world_size', default=None, type=int, help='Total number of GPUs')

    args = parser.parse_args()

    if not args.world_size:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size

    main(world_size)
```

The `init_process_group` function essentially initiates a group of processes
that can communicate with each other through the specified backend. Once this
group is initiated, you can start using collective communication routines like
broadcast, reduce, etc., or point-to-point communication routines like send and
receive.

Below, I break down each input argument and the environmental variables:

### Input Arguments

1. **`backend`**: Specifies the communication backend to use. For NVIDIA GPUs,
   "nccl" is commonly used as it is optimized for such devices. Other options
   include "gloo" and "mpi". The backend establishes how tensors will be sent
   and received.

2. **`rank`**: This is a unique identifier assigned to each process in a
   distributed setting. The `rank` is crucial for knowing which process sends or
   receives information from which other processes.

3. **`world_size`**: This is the total number of processes involved in the
   communication. Essentially, it's the size of the communication group. For
   example, if you have 4 GPUs, the world_size would generally be 4.

4. **`init_method`**: This is an optional argument that specifies how the
   distributed backend will be initialized. The most common value for this is
   `"env://"` which indicates that the initialization should use environment
   variables (`MASTER_ADDR` and `MASTER_PORT`) to set up the distributed group.
   Alternatively, you could provide a URL to a file (e.g.,
   `"file:///path/to/store/file"`), and all processes will use this file to
   share initialization information.

    When you specify `init_method='env://'`, the backend uses environment
    variables to find the address and port of the "master" process to initiate
    the distributed group. This is especially useful when you have distributed
    systems over a network and the IP address of the master node and the port
    are dynamically set or located in some cloud configuration. This way, you
    don't have to hard-code this information, providing a flexible mechanism to
    initiate the group.

    The `init_method` complements the `MASTER_ADDR` and `MASTER_PORT`
    environment variables, providing a flexible mechanism for setting up the
    distributed group. Depending on what `init_method` is set to, the
    initialization process will look for connection information in either the
    environment variables or the specified file.

### Environmental Variables

1. **`MASTER_ADDR`**: The IP address of the machine where the master process
   runs. All other processes connect to this address to set up the distributed
   system. Default is "localhost".

2. **`MASTER_PORT`**: The port number on the master machine to which all other
   processes connect. This combined with `MASTER_ADDR` establishes the
   connection among all the processes. Default is "12356".

## Point-to-Point Communication

### [Send and Recv](https://pytorch.org/tutorials/intermediate/dist_tuto.html#id1)

A transfer of data from one process to another is called a point-to-point
communication. These are achieved through the `send` and `recv` functions or
their immediate counter-parts, `isend` and `irecv`.

```python
def run(rank: int, world_size: int) -> None:
    """Blocking point-to-point communication."""
    tensor = torch.zeros(1)
    print('Rank ', rank, ' has data ', tensor[0])
    tensor = tensor.to(rank) # in 1 node, global rank = local rank
    if rank == 0:
        tensor += 1
        # Send the tensor to processes other than 0
        for other_rank in range(1, world_size):  # Sending to all other ranks
            dist.send(tensor=tensor, dst=other_rank)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

    print('Rank ', rank, ' has data ', tensor[0])
```

Each rank effectively runs its own instance of the `run` function due to the
`mp.Process` instantiation. Here's how it works in a step-by-step manner:

1. **For Rank 0**:

    - The process with `rank=0` starts and executes `run` function.
    - The `if rank == 0:` condition is true.
    - It increments its tensor from 0 to 1.
    - It performs `dist.send` to send this tensor to ranks 1, 2, and 3.
    - At this point, it has sent the data but hasn't confirmed that the data has
      been received by other ranks.

2. **For Rank 1**:

    - A new process is spawned with `rank=1`.
    - This process runs the `run` function.
    - The `else:` clause is executed.
    - It waits to receive the tensor from `rank=0` using `dist.recv`.
    - Once received, it prints the value, confirming the data transfer.

3. **For Rank 2 and 3**:
    - Similar to `rank=1`, new processes are spawned for `rank=2` and `rank=3`.
    - They also go into the `else:` clause and wait to receive the tensor from
      `rank=0`.

The `mp.Process` initiates these separate processes, and the `dist.send` and
`dist.recv` functions handle the point-to-point data communication between these
processes. Thus, the state (tensor) of `rank=0` is successfully transferred to
ranks 1, 2, and 3.

In the above example, both processes start with a zero tensor, then process 0
increments the tensor and sends it to process 1 so that they both end up with
1.0. Notice that processes 1,2 and 3 need to allocate memory in order to store
the data it will receive.

#### What does it mean by it needs to allocate memory?

In a scenario with four processes, each process initializes its own tensor
filled with zeroes in its respective memory space. Here's how the data flows:

-   **Process 0**: Modifies its tensor to 1 and sends this updated tensor to
    Processes 1, 2, and 3.
-   **Processes 1, 2, 3**: Each has its own pre-allocated tensor initialized to
    zero. When they execute `dist.recv`, they wait for the incoming data from
    Process 0.

Upon receiving the data, each of Processes 1, 2, and 3 overwrites its initially
zero-valued tensor with the received value of 1. Each process thus ends up with
a tensor containing the value 1, but these tensors are separate instances stored
in each process's individual memory space. The operation is in-place, meaning
the pre-allocated memory for the tensors in Processes 1, 2, and 3 is directly
updated. Process 0's tensor remains at its updated value of 1 and is not
affected by the receive operations in the other processes.

> The key is that `dist.recv` performs an in-place operation, modifying the
> tensor directly. The name `tensor` refers to a location in memory, and calling
> `dist.recv` changes the value stored in that memory location for Process 1.
> After the receive operation, the tensor's value in Process 1 becomes 1,
> replacing the initial zero. This does not affect the tensor in Process 0; they
> are separate instances in separate memory spaces.

## References and Further Readings

-   <https://pytorch.org/tutorials/intermediate/dist_tuto.html>
-   <https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py>
