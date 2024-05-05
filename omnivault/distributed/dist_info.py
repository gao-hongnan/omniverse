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
