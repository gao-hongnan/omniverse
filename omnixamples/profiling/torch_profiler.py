from __future__ import annotations

import logging
import socket
import sys
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Iterable, Tuple

import torch
from torch import nn
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile, record_function

from omnivault.modules.loss import CrossEntropyLoss
from omnivault.utils.reproducibility.seed import seed_all
from omnixamples.profiling.common import GPT, General, GPTConfig, device, get_random_batch

seed_all(42, True, False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


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


def trace_handler(prof: torch.profiler.profile) -> None:
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")


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

    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

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

    if profile_memory:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    return prof  # type: ignore[no-any-return]


if __name__ == "__main__":
    gpt_small_config = GPTConfig(
        context_length=128,
        vocab_size=10_000,
        d_model=768,
        num_blocks=12,
        num_heads=12,
    )
    general = General()

    seed_all(general.seed, True, False)

    batch = get_random_batch(
        batch_size=general.batch_size,
        context_length=gpt_small_config.context_length,
        vocab_size=gpt_small_config.vocab_size,
    )

    model = GPT(gpt_small_config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    run_warmup(model=model, batch=batch, optimizer=optimizer, criterion=criterion)

    profiled = run_profiler(
        model=model,
        batch=batch,
        optimizer=optimizer,
        criterion=criterion,
        enable_backward=True,
        enable_optimizer=True,
        mixed_precision=False,
        profile_steps=5,
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        with_flops=False,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=3),
        on_trace_ready=trace_handler,
    )
