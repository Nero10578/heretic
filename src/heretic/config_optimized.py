# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Enhanced configuration class with VLLM-inspired MoE optimization settings.
This module extends the original config.py with additional settings for MoE optimizations.
"""

from typing import Dict

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from .config import DatasetSpecification, Settings


class OptimizedSettings(Settings):
    """
    Enhanced settings class with VLLM-inspired MoE optimizations.
    
    This class extends the original Settings with additional configuration
    options for MoE model optimization and performance tuning.
    """
    
    # === VLLM-Inspired MoE Optimizations ===
    
    enable_moe_optimizations: bool = Field(
        default=True,
        description="Enable VLLM-inspired MoE optimizations for faster processing.",
    )
    
    moe_block_size: int = Field(
        default=64,
        description="Block size for MoE expert alignment (GPU optimization).",
        ge=16, le=256
    )
    
    moe_fused_abliteration: bool = Field(
        default=True,
        description="Use fused kernels for MoE expert abliteration.",
    )
    
    moe_cache_experts: bool = Field(
        default=True,
        description="Cache expert computations for reuse across batches.",
    )
    
    moe_batch_experts: bool = Field(
        default=True,
        description="Process multiple experts simultaneously in batches.",
    )
    
    moe_chunk_size: int = Field(
        default=4096,
        description="Chunk size for processing large MoE models.",
        ge=512, le=16384
    )
    
    moe_max_experts_per_batch: int = Field(
        default=8,
        description="Maximum number of experts to process in a single batch.",
        ge=1, le=64
    )
    
    moe_memory_efficient: bool = Field(
        default=True,
        description="Use memory-efficient processing for large MoE models.",
    )
    
    # === Advanced MoE Settings ===
    
    use_norm_preserving_abliteration: bool = Field(
        default=False,
        description="Use norm-preserving biprojected abliteration technique instead of standard abliteration.",
    )
    
    abliteration_scale_factor: float = Field(
        default=1.0,
        description="Scaling factor for abliteration strength (alpha parameter in norm-preserving abliteration).",
        ge=0.1, le=10.0
    )
    
    moe_auto_block_size: bool = Field(
        default=True,
        description="Enable automatic block size optimization.",
    )
    
    moe_min_block_size: int = Field(
        default=16,
        description="Minimum block size for automatic optimization.",
        ge=8, le=128
    )
    
    moe_max_block_size: int = Field(
        default=256,
        description="Maximum block size for automatic optimization.",
        ge=64, le=512
    )
    
    moe_load_balancing: bool = Field(
        default=True,
        description="Enable expert load balancing.",
    )
    
    moe_load_balance_threshold: float = Field(
        default=0.8,
        description="Threshold for expert load balancing (0.0-1.0).",
        ge=0.0, le=1.0
    )
    
    moe_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring.",
    )
    
    # === GPU-Specific Optimizations ===
    
    moe_cuda_optimizations: bool = Field(
        default=True,
        description="Enable CUDA-specific optimizations.",
    )
    
    moe_mixed_precision: bool = Field(
        default=True,
        description="Enable mixed precision processing.",
    )
    
    moe_use_tensor_cores: bool = Field(
        default=True,
        description="Enable tensor cores utilization.",
    )
    
    moe_memory_pool_size: int = Field(
        default=1024,
        description="Memory pool size for MoE operations (in MB).",
        ge=128, le=8192
    )
    
    # === Debug and Development Settings ===
    
    moe_verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging for MoE operations.",
    )
    
    moe_profiling: bool = Field(
        default=False,
        description="Enable performance profiling.",
    )
    
    moe_save_stats: bool = Field(
        default=False,
        description="Save performance statistics to file.",
    )
    
    moe_stats_file: str = Field(
        default="moe_performance_stats.json",
        description="File path for saving performance statistics.",
    )
    
    moe_validation: bool = Field(
        default=True,
        description="Enable validation of MoE operations.",
    )
    
    # === Compatibility Settings ===
    
    moe_fallback_enabled: bool = Field(
        default=True,
        description="Fallback to original implementation if optimization fails.",
    )
    
    moe_max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for MoE operations.",
        ge=1, le=10
    )
    
    moe_operation_timeout: int = Field(
        default=300,
        description="Timeout for MoE operations (in seconds).",
        ge=30, le=3600
    )
    
    moe_compatibility_mode: bool = Field(
        default=False,
        description="Enable compatibility mode for older models.",
    )
    
    def validate_moe_settings(self) -> Dict[str, str]:
        """
        Validate MoE settings and return any issues found.
        
        Returns:
            Dictionary of validation errors (empty if all settings are valid)
        """
        errors = {}
        
        # Check block size settings
        if self.moe_min_block_size >= self.moe_max_block_size:
            errors["block_size"] = "moe_min_block_size must be less than moe_max_block_size"
        
        if self.moe_block_size < self.moe_min_block_size or self.moe_block_size > self.moe_max_block_size:
            errors["block_size_range"] = f"moe_block_size must be between {self.moe_min_block_size} and {self.moe_max_block_size}"
        
        # Check memory settings
        if self.moe_memory_pool_size < 128:
            errors["memory_pool"] = "moe_memory_pool_size must be at least 128 MB"
        
        # Check batch settings
        if self.moe_max_experts_per_batch < 1:
            errors["batch_size"] = "moe_max_experts_per_batch must be at least 1"
        
        # Check chunk size
        if self.moe_chunk_size < 512:
            errors["chunk_size"] = "moe_chunk_size must be at least 512"
        
        # Check scaling factor
        if self.abliteration_scale_factor <= 0:
            errors["scale_factor"] = "abliteration_scale_factor must be positive"
        
        # Check load balance threshold
        if not 0.0 <= self.moe_load_balance_threshold <= 1.0:
            errors["load_balance"] = "moe_load_balance_threshold must be between 0.0 and 1.0"
        
        return errors
    
    def get_optimal_block_size(self, num_tokens: int, hidden_dim: int) -> int:
        """
        Get optimal block size based on model characteristics and settings.
        
        Args:
            num_tokens: Number of tokens to process
            hidden_dim: Hidden dimension size
        
        Returns:
            Optimal block size
        """
        if not self.moe_auto_block_size:
            return self.moe_block_size
        
        # Calculate optimal block size based on model characteristics
        candidate_blocks = [16, 32, 64, 128, 256]
        
        # Filter by min/max settings
        valid_blocks = [
            b for b in candidate_blocks 
            if self.moe_min_block_size <= b <= self.moe_max_block_size
        ]
        
        # Filter by model dimensions
        valid_blocks = [b for b in valid_blocks if b <= num_tokens and b <= hidden_dim]
        
        if not valid_blocks:
            # Fallback to configured block size
            return self.moe_block_size
        
        # Choose based on model size
        if num_tokens >= 1024 and hidden_dim >= 1024:
            # Large models: use larger blocks
            return max(valid_blocks)
        elif num_tokens >= 256 and hidden_dim >= 256:
            # Medium models: use medium blocks
            return valid_blocks[len(valid_blocks) // 2]
        else:
            # Small models: use smaller blocks
            return min(valid_blocks)
    
    def should_enable_cuda_optimizations(self) -> bool:
        """Determine if CUDA optimizations should be enabled."""
        import torch
        
        return (
            self.moe_cuda_optimizations and 
            torch.cuda.is_available() and
            torch.cuda.get_device_capability()[0] >= 7  # Volta architecture or newer
        )
    
    def should_use_mixed_precision(self) -> bool:
        """Determine if mixed precision should be used."""
        import torch
        
        return (
            self.moe_mixed_precision and
            torch.cuda.is_available() and
            torch.cuda.get_device_capability()[0] >= 7  # Volta architecture or newer
        )
    
    def get_memory_pool_size_bytes(self) -> int:
        """Get memory pool size in bytes."""
        return self.moe_memory_pool_size * 1024 * 1024
    
    def is_compatibility_mode_needed(self, model_name: str) -> bool:
        """
        Determine if compatibility mode is needed for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if compatibility mode is needed
        """
        if not self.moe_compatibility_mode:
            return False
        
        # List of models that may need compatibility mode
        compatibility_models = [
            "gpt-2",
            "gpt-neo",
            "opt",
            "bloom"
        ]
        
        return any(model in model_name.lower() for model in compatibility_models)
    
    def get_performance_config(self) -> Dict[str, any]:
        """
        Get performance configuration dictionary.
        
        Returns:
            Dictionary with performance settings
        """
        return {
            "enable_optimizations": self.enable_moe_optimizations,
            "block_size": self.moe_block_size,
            "fused_abliteration": self.moe_fused_abliteration,
            "cache_experts": self.moe_cache_experts,
            "batch_experts": self.moe_batch_experts,
            "chunk_size": self.moe_chunk_size,
            "max_experts_per_batch": self.moe_max_experts_per_batch,
            "memory_efficient": self.moe_memory_efficient,
            "auto_block_size": self.moe_auto_block_size,
            "load_balancing": self.moe_load_balancing,
            "performance_monitoring": self.moe_performance_monitoring,
            "cuda_optimizations": self.should_enable_cuda_optimizations(),
            "mixed_precision": self.should_use_mixed_precision(),
            "use_tensor_cores": self.moe_use_tensor_cores,
            "memory_pool_size": self.get_memory_pool_size_bytes(),
            "verbose_logging": self.moe_verbose_logging,
            "profiling": self.moe_profiling,
            "validation": self.moe_validation,
            "fallback_enabled": self.moe_fallback_enabled,
            "max_retries": self.moe_max_retries,
            "operation_timeout": self.moe_operation_timeout,
        }
    
    def log_configuration_summary(self) -> str:
        """
        Generate a summary of the MoE configuration.
        
        Returns:
            Formatted string with configuration summary
        """
        lines = [
            "=== MoE Optimization Configuration ===",
            f"Optimizations Enabled: {self.enable_moe_optimizations}",
            f"Block Size: {self.moe_block_size}",
            f"Fused Abliteration: {self.moe_fused_abliteration}",
            f"Expert Caching: {self.moe_cache_experts}",
            f"Batch Processing: {self.moe_batch_experts}",
            f"Max Experts per Batch: {self.moe_max_experts_per_batch}",
            f"Memory Efficient: {self.moe_memory_efficient}",
            f"Auto Block Size: {self.moe_auto_block_size}",
            f"Load Balancing: {self.moe_load_balancing}",
            f"Performance Monitoring: {self.moe_performance_monitoring}",
            "",
            "=== GPU Settings ===",
            f"CUDA Optimizations: {self.should_enable_cuda_optimizations()}",
            f"Mixed Precision: {self.should_use_mixed_precision()}",
            f"Tensor Cores: {self.moe_use_tensor_cores}",
            f"Memory Pool: {self.moe_memory_pool_size} MB",
            "",
            "=== Debug Settings ===",
            f"Verbose Logging: {self.moe_verbose_logging}",
            f"Profiling: {self.moe_profiling}",
            f"Validation: {self.moe_validation}",
            f"Fallback Enabled: {self.moe_fallback_enabled}",
        ]
        
        return "\n".join(lines)


# Factory function to create appropriate settings instance
def create_settings(use_optimized: bool = True) -> Settings:
    """
    Create appropriate settings instance.
    
    Args:
        use_optimized: Whether to use optimized settings with MoE enhancements
    
    Returns:
        Settings instance
    """
    if use_optimized:
        return OptimizedSettings()
    else:
        return Settings()


# Utility function to validate and report configuration issues
def validate_configuration(settings: Settings) -> bool:
    """
    Validate configuration and report any issues.
    
    Args:
        settings: Settings instance to validate
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if isinstance(settings, OptimizedSettings):
        errors = settings.validate_moe_settings()
        
        if errors:
            print("Configuration validation errors found:")
            for key, error in errors.items():
                print(f"  {key}: {error}")
            return False
    
    return True