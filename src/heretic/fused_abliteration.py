# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
VLLM-inspired fused abliteration kernels for MoE models.
This module provides high-performance abliteration operations that process
multiple experts simultaneously, similar to VLLM's fused_moe kernels.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

from .moe_utils import MoEProcessingCache, get_optimal_block_size


class FusedMoEAbliterator:
    """
    VLLM-inspired fused abliterator for MoE models.
    
    This class implements high-performance abliteration by batching expert
    operations and using optimized memory access patterns.
    """
    
    def __init__(self, block_size: int = 64, cache_size: int = 100):
        """
        Initialize the fused abliterator.
        
        Args:
            block_size: Block size for GPU optimization
            cache_size: Maximum number of cached tensors
        """
        self.block_size = block_size
        self.cache = MoEProcessingCache(torch.device('cpu'))  # Will be updated per device
        self.cache_size = cache_size
        
        # Performance metrics
        self.stats = {
            'experts_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def set_device(self, device: torch.device):
        """Update the cache device."""
        self.cache = MoEProcessingCache(device)
    
    def abliterate_experts_fused(
        self,
        expert_weights: torch.Tensor,  # [num_experts, out_dim, in_dim]
        refusal_directions: torch.Tensor,
        layer_index: int,
        scale_factor: float,
        block_size: Optional[int] = None,
        use_norm_preserving: bool = False
    ) -> torch.Tensor:
        """
        Fused abliteration for multiple experts simultaneously.
        
        This is the core optimization inspired by VLLM's fused_moe kernels.
        Instead of processing each expert individually, we batch the operations
        for maximum GPU utilization.
        
        Args:
            expert_weights: Stacked expert weights [num_experts, out_dim, in_dim]
            refusal_directions: Refusal directions for all layers
            layer_index: Current layer index
            scale_factor: Abliteration scaling factor
            block_size: Block size for GPU optimization
            use_norm_preserving: Use norm-preserving abliteration
        
        Returns:
            Modified expert weights [num_experts, out_dim, in_dim]
        """
        if block_size is None:
            block_size = self.block_size
        
        num_experts, out_dim, in_dim = expert_weights.shape
        device = expert_weights.device
        dtype = expert_weights.dtype
        
        # Update cache device if needed
        if self.cache.device != device:
            self.set_device(device)
        
        # Get refusal direction for this layer
        if layer_index < len(refusal_directions):
            refusal_dir = refusal_directions[layer_index].to(device).to(dtype)
        else:
            # Use last available direction if layer index exceeds
            refusal_dir = refusal_directions[-1].to(device).to(dtype)
        
        # Normalize refusal direction
        refusal_normalized = F.normalize(refusal_dir, dim=0)
        
        if use_norm_preserving:
            return self._abliterate_experts_norm_preserving(
                expert_weights, refusal_normalized, scale_factor, block_size
            )
        else:
            return self._abliterate_experts_standard(
                expert_weights, refusal_normalized, scale_factor, block_size
            )
    
    def _abliterate_experts_standard(
        self,
        expert_weights: torch.Tensor,
        refusal_normalized: torch.Tensor,
        scale_factor: float,
        block_size: int
    ) -> torch.Tensor:
        """
        Standard abliteration applied to batched experts.
        
        Args:
            expert_weights: [num_experts, out_dim, in_dim]
            refusal_normalized: Normalized refusal direction [in_dim]
            scale_factor: Scaling factor
            block_size: Block size for processing
        
        Returns:
            Modified expert weights
        """
        num_experts, out_dim, in_dim = expert_weights.shape
        device = expert_weights.device
        dtype = expert_weights.dtype
        
        # Create projection matrix once for all experts
        # This is much more efficient than creating it per expert
        cache_key = f"projector_{in_dim}_{dtype}"
        projector = self.cache.get_or_create(
            cache_key,
            (in_dim, in_dim),
            dtype,
            lambda shape, dtype, device: torch.outer(
                refusal_normalized, refusal_normalized
            ).to(dtype).to(device)
        )
        
        # Reshape for batched matrix multiplication
        # [num_experts * out_dim, in_dim]
        weights_flat = expert_weights.view(num_experts * out_dim, in_dim)
        
        # Process in blocks to optimize memory usage
        modified_weights_flat = torch.empty_like(weights_flat)
        
        for i in range(0, num_experts * out_dim, block_size):
            end_idx = min(i + block_size, num_experts * out_dim)
            block_weights = weights_flat[i:end_idx]
            
            # Batched projection: W - scale * (projector @ W.T).T
            # This is equivalent to: W - scale * (W @ projector)
            projections = torch.matmul(block_weights, projector)
            modified_block = block_weights - scale_factor * projections
            
            modified_weights_flat[i:end_idx] = modified_block
        
        # Reshape back to expert format
        modified_weights = modified_weights_flat.view(num_experts, out_dim, in_dim)
        
        # Update stats
        self.stats['experts_processed'] += num_experts
        self.stats['batches_processed'] += math.ceil(num_experts * out_dim / block_size)
        
        return modified_weights
    
    def _abliterate_experts_norm_preserving(
        self,
        expert_weights: torch.Tensor,
        refusal_normalized: torch.Tensor,
        scale_factor: float,
        block_size: int
    ) -> torch.Tensor:
        """
        Norm-preserving abliteration applied to batched experts.
        
        This maintains the row norms of the weight matrices while
        still ablating the refusal direction.
        
        Args:
            expert_weights: [num_experts, out_dim, in_dim]
            refusal_normalized: Normalized refusal direction [in_dim]
            scale_factor: Scaling factor
            block_size: Block size for processing
        
        Returns:
            Modified expert weights with preserved norms
        """
        num_experts, out_dim, in_dim = expert_weights.shape
        device = expert_weights.device
        dtype = expert_weights.dtype
        
        # Convert to float32 for numerical stability
        weights_fp32 = expert_weights.to(torch.float32)
        refusal_fp32 = refusal_normalized.to(torch.float32)
        
        # Process in blocks
        modified_weights_fp32 = torch.empty_like(weights_fp32)
        
        for i in range(0, num_experts, block_size):
            end_idx = min(i + block_size, num_experts)
            block_weights = weights_fp32[i:end_idx]  # [block_experts, out_dim, in_dim]
            
            # Apply norm-preserving transformation to the block
            modified_block = self._apply_norm_preserving_to_block(
                block_weights, refusal_fp32, scale_factor
            )
            
            modified_weights_fp32[i:end_idx] = modified_block
        
        # Convert back to original dtype
        modified_weights = modified_weights_fp32.to(dtype)
        
        # Update stats
        self.stats['experts_processed'] += num_experts
        self.stats['batches_processed'] += math.ceil(num_experts / block_size)
        
        return modified_weights
    
    def _apply_norm_preserving_to_block(
        self,
        weights_block: torch.Tensor,  # [block_experts, out_dim, in_dim]
        refusal_normalized: torch.Tensor,  # [in_dim]
        scale_factor: float
    ) -> torch.Tensor:
        """
        Apply norm-preserving abliteration to a block of experts.
        
        Args:
            weights_block: Block of expert weights
            refusal_normalized: Normalized refusal direction
            scale_factor: Scaling factor
        
        Returns:
            Modified weights block with preserved norms
        """
        block_experts, out_dim, in_dim = weights_block.shape
        
        # Reshape for processing
        weights_flat = weights_block.view(block_experts * out_dim, in_dim)
        
        # Compute original norms
        original_norms = torch.norm(weights_flat, dim=1, keepdim=True)
        
        # Compute projection onto refusal direction
        projection_coeffs = torch.matmul(weights_flat, refusal_normalized)
        projections = torch.outer(projection_coeffs, refusal_normalized)
        
        # Apply abliteration
        weights_directional = weights_flat - scale_factor * projections
        
        # Compute new norms
        new_norms = torch.norm(weights_directional, dim=1, keepdim=True)
        
        # Avoid division by zero
        new_norms = torch.clamp(new_norms, min=1e-8)
        
        # Preserve original norms
        weights_preserved = weights_directional * (original_norms / new_norms)
        
        # Reshape back
        return weights_preserved.view(block_experts, out_dim, in_dim)
    
    def abliterate_shared_experts(
        self,
        shared_weights: torch.Tensor,
        refusal_directions: torch.Tensor,
        layer_index: int,
        scale_factor: float,
        use_norm_preserving: bool = False
    ) -> torch.Tensor:
        """
        Optimized abliteration for shared experts.
        
        Shared experts are typically smaller and can be processed more efficiently.
        
        Args:
            shared_weights: Shared expert weights [out_dim, in_dim]
            refusal_directions: Refusal directions for all layers
            layer_index: Current layer index
            scale_factor: Scaling factor
            use_norm_preserving: Use norm-preserving abliteration
        
        Returns:
            Modified shared expert weights
        """
        device = shared_weights.device
        dtype = shared_weights.dtype
        
        # Get refusal direction
        if layer_index < len(refusal_directions):
            refusal_dir = refusal_directions[layer_index].to(device).to(dtype)
        else:
            refusal_dir = refusal_directions[-1].to(device).to(dtype)
        
        # Normalize refusal direction
        refusal_normalized = F.normalize(refusal_dir, dim=0)
        
        if use_norm_preserving:
            return self._apply_norm_preserving_single(
                shared_weights, refusal_normalized, scale_factor
            )
        else:
            return self._apply_standard_abliteration_single(
                shared_weights, refusal_normalized, scale_factor
            )
    
    def _apply_standard_abliteration_single(
        self,
        weights: torch.Tensor,
        refusal_normalized: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """Apply standard abliteration to a single weight matrix."""
        projector = torch.outer(refusal_normalized, refusal_normalized).to(weights.dtype)
        projection = torch.matmul(projector, weights.T).T
        return weights - scale_factor * projection
    
    def _apply_norm_preserving_single(
        self,
        weights: torch.Tensor,
        refusal_normalized: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """Apply norm-preserving abliteration to a single weight matrix."""
        # Convert to float32 for numerical stability
        weights_fp32 = weights.to(torch.float32)
        refusal_fp32 = refusal_normalized.to(torch.float32)
        
        # Compute original norms
        original_norms = torch.norm(weights_fp32, dim=1, keepdim=True)
        
        # Compute projection onto refusal direction
        projection_coeffs = torch.matmul(weights_fp32, refusal_fp32)
        projections = torch.outer(projection_coeffs, refusal_fp32)
        
        # Apply abliteration
        weights_directional = weights_fp32 - scale_factor * projections
        
        # Compute new norms
        new_norms = torch.norm(weights_directional, dim=1, keepdim=True)
        new_norms = torch.clamp(new_norms, min=1e-8)
        
        # Preserve original norms
        weights_preserved = weights_directional * (original_norms / new_norms)
        
        return weights_preserved.to(weights.dtype)
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        cache_stats = self.cache.get_cache_stats()
        
        return {
            **self.stats,
            'cache_stats': cache_stats,
            'experts_per_batch': (
                self.stats['experts_processed'] / max(1, self.stats['batches_processed'])
            )
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'experts_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def clear_cache(self):
        """Clear the processing cache."""
        self.cache.clear()


class BatchedAbliterationHook:
    """
    Enhanced abliteration hook that uses batched processing for MoE models.
    
    This hook integrates with the existing heretic infrastructure but
    provides VLLM-inspired optimizations for MoE models.
    """
    
    def __init__(
        self,
        model,
        refusal_directions: torch.Tensor,
        direction_index: Optional[float],
        parameters: dict,
        good_residuals: Optional[torch.Tensor] = None,
        bad_residuals: Optional[torch.Tensor] = None,
        enable_optimizations: bool = True
    ):
        """
        Initialize the batched abliteration hook.
        
        Args:
            model: The model instance
            refusal_directions: Refusal directions tensor
            direction_index: Direction index for interpolation
            parameters: Abliteration parameters
            good_residuals: Good residual activations
            bad_residuals: Bad residual activations
            enable_optimizations: Enable VLLM-inspired optimizations
        """
        self.model = model
        self.refusal_directions = refusal_directions
        self.direction_index = direction_index
        self.parameters = parameters
        self.good_residuals = good_residuals
        self.bad_residuals = bad_residuals
        self.enable_optimizations = enable_optimizations
        
        # Initialize fused abliterator if optimizations are enabled
        if enable_optimizations and hasattr(model.settings, 'enable_moe_optimizations') and model.settings.enable_moe_optimizations:
            self.fused_abliterator = FusedMoEAbliterator(
                block_size=getattr(model.settings, 'moe_block_size', 64)
            )
        else:
            self.fused_abliterator = None
        
        self.hooks = []
        self.use_norm_preserving = getattr(model.settings, 'use_norm_preserving_abliteration', False)
        self.scale_factor = getattr(model.settings, 'abliteration_scale_factor', 1.0)
        
        # Pre-compute mean directions for projection if needed
        if self.use_norm_preserving and good_residuals is not None and bad_residuals is not None:
            self.harmful_mean = bad_residuals.mean(dim=0)
            self.harmless_mean = good_residuals.mean(dim=0)
        else:
            self.harmful_mean = None
            self.harmless_mean = None
    
    def __enter__(self):
        """Register hooks for all layers."""
        for layer_index in range(len(self.model.get_layers())):
            layer = self.model.get_layers()[layer_index]
            
            # Hook for attention output projection
            if hasattr(layer.self_attn, 'o_proj'):
                hook = layer.self_attn.o_proj.register_forward_hook(
                    self.make_hook_fn(layer_index, 'attn.o_proj')
                )
                self.hooks.append(hook)
            
            # Hook for MLP down projection
            if hasattr(layer.mlp, 'down_proj'):
                hook = layer.mlp.down_proj.register_forward_hook(
                    self.make_hook_fn(layer_index, 'mlp.down_proj')
                )
                self.hooks.append(hook)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def make_hook_fn(self, layer_index: int, component: str):
        """Create a hook function for the given layer and component."""
        def hook_fn(module, input, output):
            # Skip if optimizations are disabled or not available
            if not self.enable_optimizations or self.fused_abliterator is None:
                return self._apply_standard_hook(layer_index, component, output)
            
            # Get parameters for this component
            if component not in self.parameters:
                return output
            
            params = self.parameters[component]
            
            # Calculate distance from max weight position
            distance = abs(layer_index - params.max_weight_position)
            
            # Skip if too far
            if distance > params.min_weight_distance:
                return output
            
            # Calculate interpolation weight
            weight = params.max_weight + (distance / params.min_weight_distance) * (
                params.min_weight - params.max_weight
            )
            
            # Get refusal direction for this layer
            layer_refusal_direction = self._get_layer_refusal_direction(layer_index)
            
            # Apply optimized abliteration based on component type
            if "batched" in component:
                return self._apply_batched_expert_abliteration(
                    output, layer_refusal_direction, weight * self.scale_factor
                )
            elif "shared" in component:
                return self._apply_shared_expert_abliteration(
                    output, layer_refusal_direction, weight * self.scale_factor
                )
            else:
                return self._apply_standard_hook(layer_index, component, output)
        
        return hook_fn
    
    def _get_layer_refusal_direction(self, layer_index: int) -> torch.Tensor:
        """Get the refusal direction for a specific layer."""
        if self.direction_index is None:
            return self.refusal_directions[layer_index + 1]
        else:
            weight_idx, idx = math.modf(self.direction_index + 1)
            return F.normalize(
                self.refusal_directions[int(idx)].lerp(
                    self.refusal_directions[int(idx) + 1],
                    weight_idx,
                ),
                p=2,
                dim=0,
            )
    
    def _apply_batched_expert_abliteration(
        self,
        output: torch.Tensor,
        refusal_direction: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """Apply abliteration to batched expert outputs."""
        device = output.device
        dtype = output.dtype
        
        # Reshape for batched processing
        original_shape = output.shape
        if len(output.shape) == 3:
            output_reshaped = output.view(-1, output.shape[-1])
        else:
            output_reshaped = output
        
        # Apply fused abliteration
        refusal_normalized = F.normalize(refusal_direction.to(device).to(dtype), dim=0)
        projector = torch.outer(refusal_normalized, refusal_normalized).to(dtype)
        
        # Batched projection
        projection = torch.matmul(output_reshaped, projector.T)
        output_modified = output_reshaped - scale_factor * projection
        
        # Reshape back
        if len(original_shape) == 3:
            return output_modified.view(original_shape)
        else:
            return output_modified
    
    def _apply_shared_expert_abliteration(
        self,
        output: torch.Tensor,
        refusal_direction: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """Apply abliteration to shared expert outputs."""
        # Similar to batched but optimized for shared experts
        return self._apply_batched_expert_abliteration(output, refusal_direction, scale_factor)
    
    def _apply_standard_hook(self, layer_index: int, component: str, output: torch.Tensor) -> torch.Tensor:
        """Apply standard abliteration (fallback)."""
        # This would implement the original heretic abliteration logic
        # For now, return output unchanged
        return output