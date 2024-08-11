from __future__ import annotations

import itertools
import logging
import sys
from contextlib import nullcontext
from timeit import default_timer
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from rich.pretty import pprint
from torch import nn
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from omnivault.modules.loss import CrossEntropyLoss
from omnivault.utils.reproducibility.seed import seed_all

seed_all(42)

import logging
import sys
from typing import Callable

import torch

F = Callable[[torch.Tensor], torch.Tensor]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def square_by_multiplication(a: torch.Tensor) -> torch.Tensor:
    return a * a


def square_by_exponentiation(a: torch.Tensor) -> torch.Tensor:
    return a**2  # type: ignore[no-any-return]


def profile_with_event(func: F, input: torch.Tensor, warmup_steps: int = 5) -> float:
    start = torch.cuda.Event(enable_timing=True)  # Create a start event
    end = torch.cuda.Event(enable_timing=True)  # Create an end event

    logger.info(f"Warmup for {warmup_steps} steps to warm up the GPU")
    for _ in range(warmup_steps):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()  # Synchronize the GPU

    time_spent: float = start.elapsed_time(end)
    return time_spent


def profile_one_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: CrossEntropyLoss,
    enable_backward: bool,
    enable_optimizer: bool,
    mixed_precision: bool = False,
) -> None:
    context = torch.autocast("cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
    inputs, targets = batch[0], batch[1]  # typically this doesn't need to be under context.
    with context:  # type: ignore[attr-defined]
        with record_function(name="forward_pass"):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if enable_backward:
            with record_function(name="backward_pass"):
                loss.backward()
            if enable_optimizer:
                with record_function(name="optimizer"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)


def run_warmup(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: CrossEntropyLoss,
) -> None:
    inputs, targets = batch[0], batch[1]
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def run_profiler(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: CrossEntropyLoss,
    enable_backward: bool,
    enable_optimizer: bool,
    mixed_precision: bool = False,
    profile_steps: int = 5,
    *,
    activities: Iterable[ProfilerActivity] | None = None,
    profile_memory: bool = False,
    with_stack: bool = True,
    record_shapes: bool = True,
    with_flops: bool = False,
    **profile_kwargs: Any,
) -> torch.profiler.profile:
    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    if profile_memory:
        logger.warning(
            "`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True"
        )
    with_stack = with_stack or profile_memory
    record_shapes = record_shapes or profile_memory

    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    with profile(
        activities=activities,  # [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        experimental_config=experimental_config,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
        **profile_kwargs,
    ) as prof:
        for _ in range(profile_steps):
            profile_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                criterion=criterion,
                enable_backward=enable_backward,
                enable_optimizer=enable_optimizer,
                mixed_precision=mixed_precision,
            )
            prof.step()

    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages().table(max_name_column_width=120, sort_by="cpu_time_total", row_limit=50))
    print(prof.key_averages().table(max_name_column_width=300, sort_by="cuda_time_total", row_limit=50))

    return prof


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()
