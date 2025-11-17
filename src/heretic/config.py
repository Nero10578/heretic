# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Configuration for heretic with MoE optimization settings.
"""

from typing import Optional
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings


class DatasetSpecification(BaseModel):
    dataset: str
    split: str = "train"


class Settings(BaseSettings):
    """Main settings class for heretic with MoE optimization support."""
    
    # MoE optimization settings
    moe_block_size: int = Field(
        default=64,
        description="Block size for MoE operations (tuned per GPU)",
        ge=16,
        le=512,
    )
    
    moe_enable_optimizations: bool = Field(
        default=False,
        description="Enable VLLM-style MoE optimizations",
    )
    
    moe_enable_profiling: bool = Field(
        default=False,
        description="Enable MoE performance profiling",
    )
    
    moe_use_quantized_kernels: bool = Field(
        default=True,
        description="Use specialized kernels for quantized models",
    )
    
    moe_max_batch_size: int = Field(
        default=1024,
        description="Maximum batch size for MoE processing",
        ge=1,
    )
    
    # Existing settings would be here in the original config
    # This is a minimal implementation to fix the import error
    model: str = Field(default="microsoft/DialoGPT-medium", description="Model name or path")
    batch_size: int = Field(default=0, description="Batch size (0 for auto)")
    max_batch_size: int = Field(default=1024, description="Maximum batch size")
    n_trials: int = Field(default=100, description="Number of optimization trials")
    n_startup_trials: int = Field(default=20, description="Number of startup trials")
    max_kl_divergence: float = Field(default=1.0, description="Maximum KL divergence")
    use_norm_preserving_abliteration: bool = Field(default=False, description="Use norm-preserving abliteration")
    abliteration_scale_factor: float = Field(default=1.0, description="Abliteration scale factor")
    system_prompt: str = Field(default="You are a helpful assistant.", description="System prompt")
    evaluate_model: Optional[str] = Field(default=None, description="Model to evaluate")
    
    # Dataset specifications
    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(dataset="Anthropic/hh-rlhf", split="train"),
        description="Good prompts dataset"
    )
    bad_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(dataset="Anthropic/hh-rlhf", split="train"),
        description="Bad prompts dataset"
    )
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def add_moe_optimization_settings(config_class):
    """
    Add MoE optimization settings to the configuration class.
    
    This function extends the existing configuration with new fields for
    MoE performance optimizations.
    """
    
    # Add MoE optimization fields
    moe_block_size = Field(
        default=64,
        description="Block size for MoE operations (tuned per GPU)",
        ge=16,
        le=512,
    )
    
    moe_enable_optimizations = Field(
        default=False,
        description="Enable VLLM-style MoE optimizations",
    )
    
    moe_enable_profiling = Field(
        default=False,
        description="Enable MoE performance profiling",
    )
    
    moe_use_quantized_kernels = Field(
        default=True,
        description="Use specialized kernels for quantized models",
    )
    
    moe_max_batch_size = Field(
        default=1024,
        description="Maximum batch size for MoE processing",
        ge=1,
    )
    
    # Add the new fields to the class
    if not hasattr(config_class, 'moe_block_size'):
        config_class.moe_block_size = moe_block_size
    if not hasattr(config_class, 'moe_enable_optimizations'):
        config_class.moe_enable_optimizations = moe_enable_optimizations
    if not hasattr(config_class, 'moe_enable_profiling'):
        config_class.moe_enable_profiling = moe_enable_profiling
    if not hasattr(config_class, 'moe_use_quantized_kernels'):
        config_class.moe_use_quantized_kernels = moe_use_quantized_kernels
    if not hasattr(config_class, 'moe_max_batch_size'):
        config_class.moe_max_batch_size = moe_max_batch_size
    
    return config_class
