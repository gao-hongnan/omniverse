from __future__ import annotations

from contextlib import nullcontext
from timeit import default_timer
from typing import List, Literal, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field

from omnivault.modules.loss import CrossEntropyLoss
from omnixamples.profiler.common import GPT


class ProfilingResults(BaseModel):
    computation: Literal["forward", "backward", "forward_backward"] = Field(..., description="Type of computation")
    times: List[float] = Field(..., description="Raw list of measured times")
    mean_time: float = Field(..., description="Mean execution time")
    median_time: float = Field(..., description="Median execution time")
    std_dev: float = Field(..., description="Standard deviation of execution times")
    min_time: float = Field(..., description="Minimum execution time")
    max_time: float = Field(..., description="Maximum execution time")
    total_time: float = Field(..., description="Total execution time")
    profile_steps: int = Field(..., description="Number of profiling runs")


def profile_model(
    model: GPT,
    batch: Tuple[torch.Tensor, torch.Tensor],
    profile_steps: int,
    computation: Literal["forward", "backward", "forward_backward"],
    warmup_steps: int | None = None,
    mixed_precision: bool = False,
) -> ProfilingResults:
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    mixed_context = torch.autocast(device.type, dtype=dtype) if mixed_precision else nullcontext()
    criterion = CrossEntropyLoss()
    inputs, targets = batch[0], batch[1]

    with mixed_context:  # type: ignore[attr-defined]
        if warmup_steps:
            for _ in range(warmup_steps):
                logits = model(inputs)
                loss = criterion(logits, targets)
                if computation in ["backward", "forward_backward"]:
                    loss.backward()
                torch.cuda.synchronize()

        times = np.zeros(profile_steps)

        for step in range(profile_steps):
            if computation == "forward":
                start = default_timer()
                logits = model(inputs)
                loss = criterion(logits, targets)
            elif computation == "backward":
                logits = model(inputs)
                loss = criterion(logits, targets)
                torch.cuda.synchronize()
                start = default_timer()
                loss.backward()
            elif computation == "forward_backward":
                start = default_timer()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
            else:
                raise ValueError(f"Invalid computation: {computation}")

            torch.cuda.synchronize()
            end = default_timer()

            time = end - start
            times[step] = time

    return ProfilingResults(
        computation=computation,
        times=times.tolist(),
        mean_time=float(np.mean(times)),
        median_time=float(np.median(times)),
        std_dev=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        total_time=float(np.sum(times)),
        profile_steps=profile_steps,
    )
