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
