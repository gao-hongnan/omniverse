import multiprocessing
import os
import socket
import warnings

import torch
import torch.distributed


def distributed_available() -> None:
    """
    Check if distributed training is available and raises error
    otherwise.
    """
    if not torch.distributed.is_available():
        raise RuntimeError(
            "Distributed training is not available. "
            "Please check that PyTorch is built with distributed support and "
            "the necessary distributed packages are installed."
        )


def distributed_initialized() -> None:
    """
    Check if distributed training is initialized and raises error otherwise.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Distributed training is not initialized. "
            "Please ensure that `torch.distributed.init_process_group()` is called "
            "before using distributed training functions."
        )


def get_world_size() -> int:
    """
    Return the total number of processes participating in the distributed training.

    This function determines the global size of the process group, which represents
    the total number of processes across all nodes involved in the distributed
    training.

    Example
    -------
    -   If the training is running on a single node with 4 processes, the function
        will return 4.
    -   If the training is running on 2 nodes with 4 processes per node, the
        function will return 8 (2 nodes * 4 processes per node).

    Note
    ----
    If we are using `torchrun`, then we can also obtain the world size using
    `os.environ["WORLD_SIZE"]`.
    """
    distributed_available()
    distributed_initialized()
    return torch.distributed.get_world_size()


def get_local_world_size() -> int:
    """
    Return the number of processes per node participating in the distributed training.

    This function determines the local size of the process group, which represents
    the number of processes running on the same node.

    Example
    -------
    -   If the training is running on a single node with 4 processes, the function
        will return 4.
    -   If the training is running on 2 nodes with 4 processes per node, the
        function will return 4 (4 processes per node).

    Note
    ----
    If we are using `torchrun`, then we can also obtain the local world size using
    `os.environ["LOCAL_WORLD_SIZE"]`.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        warnings.warn("CUDA is not available. Returning total number of CPU cores.", stacklevel=2)
        return multiprocessing.cpu_count()


def get_global_rank() -> int:
    """
    Return the rank of the current process across all nodes.

    This function determines the global rank of the current process, which
    represents the index of the current process across all nodes involved in
    the distributed training.

    Example
    -------
    -   If the training is running on a single node with 4 processes, the function
        will return the rank of the process on that node.
    -   If the training is running on 2 nodes with 4 processes per node, the
        function will return the rank of the process across all nodes.

    Note
    ----
    If we are using `torchrun`, then we can also obtain the global rank using
    `os.environ["RANK"]`.
    """
    distributed_available()
    distributed_initialized()
    return torch.distributed.get_rank()


def get_local_rank() -> int:
    """
    Return the rank of the current process on the current node.

    This function determines the local rank of the current process, which
    represents the index of the current process on the current node.

    Example
    -------
    -   If the training is running on a single node with 4 processes, the function
        will return the rank of the process on that node.
    -   If the training is running on 2 nodes with 4 processes per node, the
        function will return the rank of the process on the current node.

    Note
    ----
    If we are using `torchrun`, then we can also obtain the local rank using
    `os.environ["LOCAL_RANK"]`.
    """
    distributed_available()
    distributed_initialized()
    return get_global_rank() % get_local_world_size()


def is_master_rank() -> bool:
    """
    Check if the current process is the master process.

    The master process is the process with rank 0 across all nodes involved
    in the distributed training.
    """
    return get_global_rank() == 0


def synchronize_and_barrier() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.distributed.barrier()


def teardown() -> None:
    torch.distributed.destroy_process_group()


def is_free_port(port: int) -> bool:
    """Checks if a given port is free on all relevant interfaces.

    References
    ----------
    -   https://github.com/serend1p1ty/core-pytorch-utils
    """
    ips = ["localhost", "127.0.0.1"]
    try:
        ips.extend(socket.gethostbyname_ex(socket.gethostname())[-1])
    except socket.gaierror:
        warnings.warn("Failed to get all IPs for the current host. Only using localhost", stacklevel=2)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return all(sock.connect_ex((ip, port)) != 0 for ip in ips)


def find_free_port() -> int:
    """Finds a free port by binding to port 0 and then checking if it's truly free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port: int = sock.getsockname()[1]
        if is_free_port(port):
            return port
        else:
            return find_free_port()  # Recursively search if the first found port is not free


def get_hostname() -> str:
    """Get the hostname of the machine."""
    return socket.gethostname()


def get_process_id() -> int:
    """Get the process id of the current process."""
    return os.getpid()
