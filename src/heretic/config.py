# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Configuration for heretic with MoE optimization settings.
"""

from typing import Optional
from pydantic import Field


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
        default=True,
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
