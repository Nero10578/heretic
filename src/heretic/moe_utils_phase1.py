# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Phase 1 MoE optimization utilities for heretic.
This module provides essential expert batching optimizations for immediate performance gains.
"""

import torch
from typing import Tuple, Optional, List


def batch_expert_weights(expert_weights_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Batch expert weights for efficient processing.
    
    This is the core Phase 1 optimization - instead of processing experts
    individually, we batch them together for vectorized operations.
    
    Args:
        expert_weights_list: List of expert weight tensors [out_dim, in_dim]
    
    Returns:
        Batched expert weights [num_experts, out_dim, in_dim]
    """
    if not expert_weights_list:
        raise ValueError("expert_weights_list cannot be empty")
    
    # Ensure all weights have the same shape
    shape = expert_weights_list[0].shape
    for i, weights in enumerate(expert_weights_list):
        if weights.shape != shape:
            raise ValueError(f"Expert {i} has shape {weights.shape}, expected {shape}")
    
    # Stack along new dimension for batched processing
    return torch.stack(expert_weights_list, dim=0)


def unbatch_expert_weights(batched_weights: torch.Tensor) -> List[torch.Tensor]:
    """
    Unbatch expert weights back to list format.
    
    Args:
        batched_weights: [num_experts, out_dim, in_dim]
    
    Returns:
        List of expert weight tensors
    """
    return [batched_weights[i] for i in range(batched_weights.size(0))]


def detect_moe_layer(layer) -> bool:
    """
    Detect if a layer is an MoE layer.
    
    Args:
        layer: Transformer layer to check
    
    Returns:
        True if this is an MoE layer
    """
    # Check for common MoE architectures
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        
        # GLM MoE with experts and shared experts
        if hasattr(mlp, 'experts') and hasattr(mlp, 'shared_experts'):
            return True
        
        # Standard MoE (Mixtral, Qwen3, etc.)
        if hasattr(mlp, 'experts'):
            return True
        
        # Phi-3.5-MoE style
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
            return True
        
        # gpt-oss MoE style
        if hasattr(mlp, 'experts') and hasattr(mlp.experts, 'down_proj'):
            return True
    
    return False


def get_expert_weights_from_layer(layer) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract expert weights from an MoE layer.
    
    Args:
        layer: Transformer layer with MoE structure
    
    Returns:
        Tuple of (expert_weights_list, shared_expert_weights)
    """
    expert_weights = []
    shared_expert_weights = None
    
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        
        # GLM MoE models
        if hasattr(mlp, 'experts') and hasattr(mlp, 'shared_experts'):
            # Collect expert weights
            for expert in mlp.experts:
                if hasattr(expert, 'down_proj'):
                    expert_weights.append(expert.down_proj.weight)
            
            # Get shared expert weights
            if hasattr(mlp.shared_experts, 'down_proj'):
                shared_expert_weights = mlp.shared_experts.down_proj.weight
        
        # Standard MoE models (Mixtral, Qwen3, etc.)
        elif hasattr(mlp, 'experts'):
            for expert in mlp.experts:
                if hasattr(expert, 'down_proj'):
                    expert_weights.append(expert.down_proj.weight)
                elif hasattr(expert, 'w2'):  # Alternative naming
                    expert_weights.append(expert.w2.weight)
        
        # Phi-3.5-MoE style
        elif hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
            for expert in layer.block_sparse_moe.experts:
                if hasattr(expert, 'w2'):
                    expert_weights.append(expert.w2.weight)
        
        # gpt-oss MoE style
        elif hasattr(mlp, 'experts') and hasattr(mlp.experts, 'down_proj'):
            # This is a 3D tensor containing all expert weights
            expert_weights.append(mlp.experts.down_proj)
    
    return expert_weights, shared_expert_weights


def apply_abliteration_to_batched_experts(
    batched_expert_weights: torch.Tensor,
    refusal_direction: torch.Tensor,
    scale_factor: float
) -> torch.Tensor:
    """
    Apply abliteration to batched expert weights.
    
    This is the core Phase 1 optimization - we apply abliteration
    to all experts simultaneously using vectorized operations.
    
    Args:
        batched_expert_weights: [num_experts, out_dim, in_dim]
        refusal_direction: [in_dim] normalized refusal direction
        scale_factor: Abliteration scaling factor
    
    Returns:
        Modified batched expert weights
    """
    num_experts, out_dim, in_dim = batched_expert_weights.shape
    device = batched_expert_weights.device
    dtype = batched_expert_weights.dtype
    
    # Normalize refusal direction
    refusal_normalized = torch.nn.functional.normalize(refusal_direction.to(device).to(dtype), dim=0)
    
    # Create projection matrix once for all experts
    # This is much more efficient than creating it per expert
    projector = torch.outer(refusal_normalized, refusal_normalized).to(dtype)
    
    # Reshape for batched matrix multiplication
    # [num_experts * out_dim, in_dim]
    weights_flat = batched_expert_weights.view(num_experts * out_dim, in_dim)
    
    # Apply abliteration to all experts simultaneously
    # W_modified = W - scale * (projector @ W.T).T
    # This is equivalent to: W_modified = W - scale * (W @ projector)
    projections = torch.matmul(weights_flat, projector)
    modified_weights_flat = weights_flat - scale_factor * projections
    
    # Reshape back to expert format
    modified_weights = modified_weights_flat.view(num_experts, out_dim, in_dim)
    
    return modified_weights


def apply_abliteration_to_shared_expert(
    shared_expert_weights: torch.Tensor,
    refusal_direction: torch.Tensor,
    scale_factor: float
) -> torch.Tensor:
    """
    Apply abliteration to shared expert weights.
    
    Args:
        shared_expert_weights: [out_dim, in_dim]
        refusal_direction: [in_dim] normalized refusal direction
        scale_factor: Abliteration scaling factor
    
    Returns:
        Modified shared expert weights
    """
    device = shared_expert_weights.device
    dtype = shared_expert_weights.dtype
    
    # Normalize refusal direction
    refusal_normalized = torch.nn.functional.normalize(refusal_direction.to(device).to(dtype), dim=0)
    
    # Create projection matrix
    projector = torch.outer(refusal_normalized, refusal_normalized).to(dtype)
    
    # Apply abliteration
    projection = torch.matmul(projector, shared_expert_weights.T).T
    modified_weights = shared_expert_weights - scale_factor * projection
    
    return modified_weights


class Phase1MoEOptimizer:
    """
    Phase 1 MoE optimizer with essential optimizations.
    
    This class provides the core Phase 1 optimizations:
    - Expert batching for vectorized processing
    - Simple caching for memory efficiency
    - Basic performance monitoring
    """
    
    def __init__(self):
        """Initialize Phase 1 MoE optimizer."""
        self.stats = {
            'experts_processed': 0,
            'batches_processed': 0,
            'layers_processed': 0,
            'shared_experts_processed': 0
        }
    
    def process_layer_matrices(self, layer, layer_index: int) -> dict:
        """
        Process layer matrices with Phase 1 optimizations.
        
        Args:
            layer: Transformer layer
            layer_index: Layer index for tracking
        
        Returns:
            Dictionary of processed matrices
        """
        matrices = {}
        
        # Standard attention processing (unchanged)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            matrices["attn.o_proj"] = [layer.self_attn.o_proj.weight]
        
        # Check if this is an MoE layer
        if detect_moe_layer(layer):
            expert_weights, shared_expert_weights = get_expert_weights_from_layer(layer)
            
            if expert_weights:
                # Batch expert weights for efficient processing
                try:
                    batched_experts = batch_expert_weights(expert_weights)
                    matrices["mlp.down_proj.batched"] = [batched_experts]
                    self.stats['experts_processed'] += len(expert_weights)
                    self.stats['batches_processed'] += 1
                except ValueError as e:
                    # Fallback to individual processing if batching fails
                    print(f"Warning: Could not batch experts at layer {layer_index}: {e}")
                    for i, expert_weight in enumerate(expert_weights):
                        matrices[f"mlp.down_proj.expert_{i}"] = [expert_weight]
            
            if shared_expert_weights is not None:
                matrices["mlp.shared_down_proj"] = [shared_expert_weights]
                self.stats['shared_experts_processed'] += 1
        else:
            # Fallback to standard MLP processing
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
                matrices["mlp.down_proj"] = [layer.mlp.down_proj.weight]
        
        self.stats['layers_processed'] += 1
        return matrices
    
    def apply_abliteration_to_matrices(
        self,
        matrices: dict,
        refusal_direction: torch.Tensor,
        scale_factor: float
    ) -> dict:
        """
        Apply abliteration to processed matrices.
        
        Args:
            matrices: Dictionary of matrices from process_layer_matrices
            refusal_direction: Refusal direction for this layer
            scale_factor: Abliteration scaling factor
        
        Returns:
            Dictionary of modified matrices
        """
        modified_matrices = {}
        
        for component, matrix_list in matrices.items():
            modified_matrix_list = []
            
            for matrix in matrix_list:
                if "batched" in component:
                    # Apply batched abliteration
                    modified_matrix = apply_abliteration_to_batched_experts(
                        matrix, refusal_direction, scale_factor
                    )
                    modified_matrix_list.append(modified_matrix)
                elif "shared" in component:
                    # Apply shared expert abliteration
                    modified_matrix = apply_abliteration_to_shared_expert(
                        matrix, refusal_direction, scale_factor
                    )
                    modified_matrix_list.append(modified_matrix)
                else:
                    # Fallback to standard abliteration for non-MoE components
                    modified_matrix = self._apply_standard_abliteration(
                        matrix, refusal_direction, scale_factor
                    )
                    modified_matrix_list.append(modified_matrix)
            
            modified_matrices[component] = modified_matrix_list
        
        return modified_matrices
    
    def _apply_standard_abliteration(
        self,
        matrix: torch.Tensor,
        refusal_direction: torch.Tensor,
        scale_factor: float
    ) -> torch.Tensor:
        """
        Apply standard abliteration to a single matrix.
        
        Args:
            matrix: Weight matrix to modify
            refusal_direction: Refusal direction
            scale_factor: Scaling factor
        
        Returns:
            Modified matrix
        """
        device = matrix.device
        dtype = matrix.dtype
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(
            refusal_direction.to(device).to(dtype), dim=0
        )
        
        # Create projection matrix
        projector = torch.outer(refusal_normalized, refusal_normalized).to(dtype)
        
        # Apply abliteration
        projection = torch.matmul(projector, matrix.T).T
        modified_matrix = matrix - scale_factor * projection
        
        return modified_matrix
    
    def get_performance_stats(self) -> dict:
        """Get Phase 1 performance statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'experts_processed': 0,
            'batches_processed': 0,
            'layers_processed': 0,
            'shared_experts_processed': 0
        }


def update_matrix_in_place(original_matrix: torch.Tensor, modified_matrix: torch.Tensor):
    """
    Update matrix in place, handling different quantization types.
    
    Args:
        original_matrix: Original matrix to update
        modified_matrix: Modified matrix with new values
    """
    if hasattr(original_matrix, 'bnb_quantized') and original_matrix.bnb_quantized:
        # Handle bitsandbytes quantization
        try:
            if hasattr(original_matrix, 'weight') and original_matrix.weight is not None:
                with torch.no_grad():
                    original_matrix.weight.data = modified_matrix
        except Exception as e:
            print(f"Warning: Failed to update quantized matrix: {e}")
    elif hasattr(original_matrix, '_data_impl') and hasattr(original_matrix._data_impl, '_value'):
        # Handle torchao quantization
        with torch.no_grad():
            original_matrix.copy_(modified_matrix)
    else:
        # Handle regular weights
        with torch.no_grad():
            original_matrix.copy_(modified_matrix)