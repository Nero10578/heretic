# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Phase 1 configuration class with essential MoE optimization settings.
This module extends the original config.py with Phase 1 optimization options.
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


class Phase1Settings(Settings):
    """
    Phase 1 settings class with essential MoE optimizations.
    
    This class extends the original Settings with Phase 1 configuration
    options for immediate performance improvements with minimal risk.
    """
    
    # === Phase 1 MoE Optimizations ===
    
    enable_phase1_optimizations: bool = Field(
        default=True,
        description="Enable Phase 1 MoE optimizations for immediate performance gains.",
    )
    
    phase1_batch_experts: bool = Field(
        default=True,
        description="Process multiple experts simultaneously in batches.",
    )
    
    phase1_memory_efficient: bool = Field(
        default=True,
        description="Use memory-efficient processing for large MoE models.",
    )
    
    phase1_max_batch_size: int = Field(
        default=16,
        description="Maximum batch size for Phase 1 optimizations.",
        ge=1, le=64
    )
    
    phase1_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring for Phase 1 optimizations.",
    )
    
    # === Advanced Phase 1 Settings ===
    
    use_norm_preserving_abliteration: bool = Field(
        default=False,
        description="Use norm-preserving biprojected abliteration technique instead of standard abliteration.",
    )
    
    abliteration_scale_factor: float = Field(
        default=1.0,
        description="Scaling factor for abliteration strength (alpha parameter in norm-preserving abliteration).",
        ge=0.1, le=10.0
    )
    
    # === Compatibility and Fallback Settings ===
    
    phase1_fallback_enabled: bool = Field(
        default=True,
        description="Fallback to original implementation if optimization fails.",
    )
    
    phase1_max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for Phase 1 operations.",
        ge=1, le=10
    )
    
    phase1_validation: bool = Field(
        default=True,
        description="Enable validation of Phase 1 operations.",
    )
    
    # === Debug Settings ===
    
    phase1_verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging for Phase 1 operations.",
    )
    
    phase1_save_stats: bool = Field(
        default=False,
        description="Save Phase 1 performance statistics to file.",
    )
    
    phase1_stats_file: str = Field(
        default="phase1_performance_stats.json",
        description="File path for saving Phase 1 performance statistics.",
    )
    
    def validate_phase1_settings(self) -> Dict[str, str]:
        """
        Validate Phase 1 settings and return any issues found.
        
        Returns:
            Dictionary of validation errors (empty if all settings are valid)
        """
        errors = {}
        
        # Check batch size settings
        if self.phase1_max_batch_size < 1:
            errors["batch_size"] = "phase1_max_batch_size must be at least 1"
        
        if self.phase1_max_batch_size > 64:
            errors["batch_size_large"] = "phase1_max_batch_size should not exceed 64 for stability"
        
        # Check scaling factor
        if self.abliteration_scale_factor <= 0:
            errors["scale_factor"] = "abliteration_scale_factor must be positive"
        
        # Check retry settings
        if self.phase1_max_retries < 1:
            errors["retries"] = "phase1_max_retries must be at least 1"
        
        return errors
    
    def get_phase1_performance_config(self) -> Dict[str, any]:
        """
        Get Phase 1 performance configuration dictionary.
        
        Returns:
            Dictionary with Phase 1 performance settings
        """
        return {
            "enable_optimizations": self.enable_phase1_optimizations,
            "batch_experts": self.phase1_batch_experts,
            "memory_efficient": self.phase1_memory_efficient,
            "max_batch_size": self.phase1_max_batch_size,
            "performance_monitoring": self.phase1_performance_monitoring,
            "fallback_enabled": self.phase1_fallback_enabled,
            "max_retries": self.phase1_max_retries,
            "validation": self.phase1_validation,
            "verbose_logging": self.phase1_verbose_logging,
        }
    
    def is_phase1_enabled(self) -> bool:
        """Check if Phase 1 optimizations are enabled."""
        return self.enable_phase1_optimizations
    
    def should_use_batch_experts(self) -> bool:
        """Check if expert batching should be used."""
        return self.enable_phase1_optimizations and self.phase1_batch_experts
    
    def should_use_memory_efficient(self) -> bool:
        """Check if memory-efficient processing should be used."""
        return self.enable_phase1_optimizations and self.phase1_memory_efficient
    
    def get_optimal_batch_size(self, suggested_size: int) -> int:
        """
        Get optimal batch size based on Phase 1 settings.
        
        Args:
            suggested_size: Suggested batch size from other sources
        
        Returns:
            Optimal batch size considering Phase 1 constraints
        """
        if not self.enable_phase1_optimizations:
            return suggested_size
        
        # Apply Phase 1 constraints
        optimal_size = min(suggested_size, self.phase1_max_batch_size)
        
        # Ensure minimum batch size
        if optimal_size < 1:
            optimal_size = 1
        
        return optimal_size
    
    def log_phase1_configuration_summary(self) -> str:
        """
        Generate a summary of the Phase 1 configuration.
        
        Returns:
            Formatted string with Phase 1 configuration summary
        """
        lines = [
            "=== Phase 1 MoE Optimization Configuration ===",
            f"Optimizations Enabled: {self.enable_phase1_optimizations}",
            f"Expert Batching: {self.phase1_batch_experts}",
            f"Memory Efficient: {self.phase1_memory_efficient}",
            f"Max Batch Size: {self.phase1_max_batch_size}",
            f"Performance Monitoring: {self.phase1_performance_monitoring}",
            "",
            "=== Compatibility Settings ===",
            f"Fallback Enabled: {self.phase1_fallback_enabled}",
            f"Max Retries: {self.phase1_max_retries}",
            f"Validation: {self.phase1_validation}",
            "",
            "=== Debug Settings ===",
            f"Verbose Logging: {self.phase1_verbose_logging}",
            f"Save Stats: {self.phase1_save_stats}",
        ]
        
        return "\n".join(lines)
    
    def get_model_specific_recommendations(self, model_name: str) -> Dict[str, any]:
        """
        Get model-specific recommendations for Phase 1 settings.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with recommended settings
        """
        model_name_lower = model_name.lower()
        
        recommendations = {}
        
        # GLM MoE models
        if "glm" in model_name_lower:
            recommendations.update({
                "phase1_max_batch_size": 8,  # Conservative for GLM
                "phase1_memory_efficient": True,
                "phase1_batch_experts": True,
            })
        
        # Mixtral models
        elif "mixtral" in model_name_lower:
            recommendations.update({
                "phase1_max_batch_size": 16,  # Standard for Mixtral
                "phase1_memory_efficient": True,
                "phase1_batch_experts": True,
            })
        
        # Large models (>70B parameters)
        elif any(size in model_name for size in ["70b", "65b", "175b"]):
            recommendations.update({
                "phase1_max_batch_size": 4,  # Very conservative for large models
                "phase1_memory_efficient": True,
                "phase1_batch_experts": True,
            })
        
        # Small models (<10B parameters)
        elif any(size in model_name for size in ["7b", "8b", "13b"]):
            recommendations.update({
                "phase1_max_batch_size": 32,  # More aggressive for small models
                "phase1_memory_efficient": False,  # Less need for memory efficiency
                "phase1_batch_experts": True,
            })
        
        # Default recommendations
        else:
            recommendations.update({
                "phase1_max_batch_size": 16,
                "phase1_memory_efficient": True,
                "phase1_batch_experts": True,
            })
        
        return recommendations


# Factory function to create appropriate settings instance
def create_phase1_settings(use_phase1: bool = True) -> Settings:
    """
    Create appropriate settings instance.
    
    Args:
        use_phase1: Whether to use Phase 1 settings with MoE enhancements
    
    Returns:
        Settings instance
    """
    if use_phase1:
        return Phase1Settings()
    else:
        return Settings()


# Utility function to validate and report Phase 1 configuration issues
def validate_phase1_configuration(settings: Settings) -> bool:
    """
    Validate Phase 1 configuration and report any issues.
    
    Args:
        settings: Settings instance to validate
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if isinstance(settings, Phase1Settings):
        errors = settings.validate_phase1_settings()
        
        if errors:
            print("Phase 1 configuration validation errors found:")
            for key, error in errors.items():
                print(f"  {key}: {error}")
            return False
    
    return True


# Utility function to apply model-specific recommendations
def apply_model_recommendations(settings: Phase1Settings, model_name: str) -> Phase1Settings:
    """
    Apply model-specific recommendations to Phase 1 settings.
    
    Args:
        settings: Phase 1 settings to modify
        model_name: Name of the model
    
    Returns:
        Modified settings with recommendations applied
    """
    recommendations = settings.get_model_specific_recommendations(model_name)
    
    # Apply recommendations (only if not explicitly set by user)
    for key, value in recommendations.items():
        if hasattr(settings, key):
            current_value = getattr(settings, key)
            # Only apply if current value is the default
            default_value = Phase1Settings.__fields__[key].default
            if current_value == default_value:
                setattr(settings, key, value)
                print(f"Applied recommendation: {key} = {value}")
    
    return settings