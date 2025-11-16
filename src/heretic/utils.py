# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import gc
from dataclasses import asdict
from importlib.metadata import version
from typing import TypeVar

import torch
import torch.distributed as dist
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_sdaa_available,
    is_xpu_available,
)
from datasets import load_dataset
from optuna import Trial
from rich.console import Console

from .config import DatasetSpecification, Settings

print = Console(highlight=False).print


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def load_prompts(specification: DatasetSpecification) -> list[str]:
    dataset = load_dataset(specification.dataset, split=specification.split)
    return list(dataset[specification.column])


T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def empty_cache():
    # Force garbage collection first
    gc.collect()
    
    # Clear device-specific caches for ALL available devices
    if torch.cuda.is_available():
        # Clear cache for ALL GPUs, not just the current one
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()  # Ensure all operations are complete
                torch.cuda.empty_cache()
                # Reset peak memory stats to get accurate measurements
                torch.cuda.reset_peak_memory_stats()
    elif is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    
    # Force garbage collection again after clearing device cache
    gc.collect()


def fsdp_empty_cache():
    """
    FSDP-aware cache clearing that handles distributed environments.
    This function ensures proper synchronization across all processes
    before clearing caches, which is critical for FSDP to work correctly.
    """
    # Only perform distributed operations if FSDP is initialized
    if dist.is_initialized():
        # Force garbage collection first
        gc.collect()
        
        # Synchronize all processes before clearing caches
        if torch.cuda.is_available():
            # Clear cache for ALL GPUs, not just the current one
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.synchronize()  # Ensure all operations are complete
                    torch.cuda.empty_cache()
                    # Reset peak memory stats to get accurate measurements
                    torch.cuda.reset_peak_memory_stats()
        elif is_xpu_available():
            torch.xpu.empty_cache()
        elif is_mlu_available():
            torch.mlu.empty_cache()
        elif is_sdaa_available():
            torch.sdaa.empty_cache()
        elif is_musa_available():
            torch.musa.empty_cache()
        
        # Force garbage collection again after clearing device cache
        gc.collect()
        
        # Synchronize all processes to ensure cache clearing is complete
        dist.barrier()
    else:
        # If not in distributed environment, use regular cache clearing
        empty_cache()


def print_memory_usage(prefix: str = ""):
    """Print current memory usage for debugging purposes"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # Convert to GB
            max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)  # Convert to GB
            
            device_prefix = f"{prefix}GPU {i}: " if prefix else f"GPU {i}: "
            print(f"[grey50]{device_prefix}Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB[/]")


def fsdp_print_memory_usage(prefix: str = ""):
    """
    FSDP-aware memory usage printing that handles distributed environments.
    This function prints memory usage for all processes in a distributed setting.
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)  # Convert to GB
                max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)  # Convert to GB
                
                device_prefix = f"{prefix}Rank {rank}, GPU {i}: " if prefix else f"Rank {rank}, GPU {i}: "
                print(f"[grey50]{device_prefix}Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {max_allocated:.2f}GB[/]")
                
                # Print total memory usage across all processes
                if i == 0:  # Only print once per rank
                    total_allocated = allocated * world_size
                    total_reserved = reserved * world_size
                    print(f"[grey50]{prefix}Total across {world_size} processes: Allocated: {total_allocated:.2f}GB, Reserved: {total_reserved:.2f}GB[/]")
    else:
        # If not in distributed environment, use regular memory printing
        print_memory_usage(prefix)


def get_trial_parameters(trial: Trial) -> dict[str, str]:
    params = {}

    direction_index = trial.user_attrs["direction_index"]
    params["direction_index"] = (
        "per layer" if (direction_index is None) else f"{direction_index:.2f}"
    )

    for component, parameters in trial.user_attrs["parameters"].items():
        for name, value in asdict(parameters).items():
            params[f"{component}.{name}"] = f"{value:.2f}"

    return params


def get_readme_intro(
    settings: Settings,
    trial: Trial,
    base_refusals: int,
    bad_prompts: list[str],
) -> str:
    model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"

    return f"""# This is a decensored version of {
        model_link
    }, made using [Heretic](https://github.com/p-e-w/heretic) v{version("heretic")}

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in get_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
| **KL divergence** | {trial.user_attrs["kl_divergence"]:.2f} | 0 *(by definition)* |
| **Refusals** | {trial.user_attrs["refusals"]}/{len(bad_prompts)} | {base_refusals}/{
        len(bad_prompts)
    } |

-----

"""
