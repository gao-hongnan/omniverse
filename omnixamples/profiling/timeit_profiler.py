from __future__ import annotations

import itertools
import logging
import sys
from contextlib import nullcontext
from timeit import default_timer
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from rich.pretty import pprint
from tqdm import tqdm

from omnivault.modules.loss import CrossEntropyLoss
from omnivault.utils.reproducibility.seed import seed_all
from omnivault.utils.torch_utils.cleanup import purge_global_scope
from omnixamples.profiling.common import GPT, General, GPTConfig, ProfilerConfig, device, get_random_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

logger.info("Device=%s", device)


class ProfilingResult(BaseModel):
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
) -> ProfilingResult:
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

    return ProfilingResult(
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


def create_profile_configs(context_length: int, vocab_size: int) -> Iterable[Tuple[str, GPTConfig, ProfilerConfig]]:
    gpt_configs: Dict[str, Dict[str, int]] = {
        "small": {"d_model": 768, "num_blocks": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "num_blocks": 24, "num_heads": 16},
    }
    computations: Tuple[Literal["forward", "backward", "forward_backward"], ...] = (
        "forward",
        "backward",
        "forward_backward",
    )
    warmup_steps: Tuple[int, ...] = (0, 1)
    mixed_precision_options: Tuple[bool, ...] = (False, True)
    profile_steps: Tuple[int, ...] = (5,)

    for (config_name, config), computation, warmup, mixed, steps in itertools.product(
        gpt_configs.items(),
        computations,
        warmup_steps,
        mixed_precision_options,
        profile_steps,
    ):
        gpt_config = GPTConfig(**config, context_length=context_length, vocab_size=vocab_size)  # type: ignore[arg-type]
        profiler_config = ProfilerConfig(
            computation=computation,
            warmup_steps=warmup,
            profile_steps=steps,
            mixed_precision=mixed,
        )
        yield config_name, gpt_config, profiler_config


def run_profile(
    device: torch.device,
    gpt_config: GPTConfig,
    profiler_config: ProfilerConfig,
    general: General,
) -> ProfilingResult:
    logger.info("Running profile with GPT config: \n%s", gpt_config.model_dump_json(indent=4))
    logger.info("Profiler config: \n%s", profiler_config.model_dump_json(indent=4))

    seed_all(general.seed, True, False)
    batch = get_random_batch(
        batch_size=general.batch_size,
        context_length=gpt_config.context_length,
        vocab_size=gpt_config.vocab_size,
    )

    gpt = GPT(config=gpt_config).to(device)

    result = profile_model(
        model=gpt,
        batch=batch,
        warmup_steps=profiler_config.warmup_steps,
        profile_steps=profiler_config.profile_steps,
        mixed_precision=profiler_config.mixed_precision,
        computation=profiler_config.computation,
    )

    logger.warning("Purging global scope variables `gpt` and `batch` to free up memory.")
    purge_global_scope(variable_name_or_names=["gpt", "batch"])
    return result


def results_to_dataframe(results: Dict[str, ProfilingResult]) -> pd.DataFrame:
    data = []
    for name, result in results.items():
        row = result.model_dump()
        row["name"] = name
        data.append(row)

    df = pd.DataFrame(data)
    columns = ["name"] + [col for col in df.columns if col != "name"]
    df = df[columns]
    return df


def main() -> Dict[str, ProfilingResult]:
    context_length = 128
    vocab_size = 10_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    general = General()

    results: Dict[str, ProfilingResult] = {}

    all_configs = list(create_profile_configs(context_length, vocab_size))

    for config_name, gpt_config, profiler_config in tqdm(all_configs, desc="Profiling Configurations"):
        key = (
            f"{config_name}_{profiler_config.computation}_"
            f"warmup_{profiler_config.warmup_steps}_"
            f"mixed_{profiler_config.mixed_precision}"
        )
        logger.info("Running profile for: %s", key)
        results[key] = run_profile(device, gpt_config, profiler_config, general)
        logger.info("Profile result: \n%s\n\n\n", results[key].model_dump_json(indent=4))

    return results


if __name__ == "__main__":
    results = main()
    pprint(results)
