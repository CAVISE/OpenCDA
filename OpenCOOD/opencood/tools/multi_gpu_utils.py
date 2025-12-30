"""
Distributed training utilities for multi-GPU and multi-node training.
"""

import os
from typing import Any, Tuple

import torch
import torch.distributed as dist


def get_dist_info() -> Tuple[int, int]:
    """
    Get the distributed process information.

    Returns
    -------
    rank : int
        Process rank within the distributed group.
        Returns 0 if not using distributed training.
    world_size : int
        Number of processes in the distributed group.
        Returns 1 if not using distributed training.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_distributed_mode(args: Any) -> None:
    """
    Initialize distributed training environment.

    Parameters
    ----------
    args : Any
        Command line arguments object that will be updated with:
            - rank (int): Process rank
            - world_size (int): Number of processes
            - gpu (int): Local GPU ID
            - distributed (bool): Whether distributed training is enabled
            - dist_backend (str): Backend for distributed training ('nccl')
            - dist_url (str): URL for distributed training setup
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master: bool) -> None:
    """
    Disable printing when not in master process.

    Parameters
    ----------
    is_master : bool
        Whether the current process is the master process.

    Notes
    -----
    This function modifies the built-in print function to only print
    from the master process, unless the 'force' keyword argument is provided.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
