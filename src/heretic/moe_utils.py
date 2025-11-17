# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
VLLM-inspired MoE optimization utilities for heretic.
This module provides token alignment and batching optimizations for MoE models.
"""

import torch
from typing import Tuple, Optional
import math


def align_tokens_for_experts(
    topk_ids: torch.Tensor,
    block_size: int = 64,
    num_experts: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VLLM-inspired token alignment for efficient expert processing.
    
    This function aligns token distribution across experts to be compatible
    with block size for matrix multiplication, similar to VLLM's moe_align_block_size.
    
    Args:
        topk_ids: [batch_size, top_k] expert indices for each token
        block_size: GPU-optimized block size (typically 32-128)
        num_experts: Total number of experts
    
    Returns:
        sorted_token_ids: Tokens sorted by expert assignment
        expert_ids: Expert ID for each block
        num_tokens_post_padded: Total tokens after padding
    """
    if num_experts is None:
        num_experts = topk_ids.max().item() + 1
    
    device = topk_ids.device
    dtype = topk_ids.dtype
    
    # Flatten topk_ids for processing
    flat_ids = topk_ids.flatten()
    
    # Calculate tokens per expert
    tokens_per_expert = torch.bincount(flat_ids, minlength=num_experts)
    
    # Calculate padding needed for block alignment
    padded_tokens_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size
    
    # Create sorted token list with padding
    sorted_tokens = []
    expert_ids_list = []
    
    for expert_id in range(num_experts):
        # Find tokens assigned to this expert
        expert_tokens = torch.where(flat_ids == expert_id)[0]
        
        if len(expert_tokens) > 0:
            # Calculate padding needed
            current_count = len(expert_tokens)
            target_count = padded_tokens_per_expert[expert_id].item()
            padding_needed = target_count - current_count
            
            if padding_needed > 0:
                # Add padding tokens (marked as -1)
                padding = torch.full((padding_needed,), -1, dtype=dtype, device=device)
                expert_tokens = torch.cat([expert_tokens, padding])
            
            sorted_tokens.append(expert_tokens)
            expert_ids_list.extend([expert_id] * target_count)
    
    if sorted_tokens:
        sorted_token_ids = torch.cat(sorted_tokens)
        expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=device)
        num_tokens_post_padded = torch.tensor([len(sorted_token_ids)], dtype=torch.int32, device=device)
    else:
        sorted_token_ids = torch.empty(0, dtype=dtype, device=device)
        expert_ids = torch.empty(0, dtype=torch.int32, device=device)
        num_tokens_post_padded = torch.tensor([0], dtype=torch.int32, device=device)
    
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def batch_expert_weights(expert_weights_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Batch expert weights for efficient processing.
    
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
    
    # Stack along new dimension
    return torch.stack(expert_weights_list, dim=0)


def unbatch_expert_weights(batched_weights: torch.Tensor) -> list[torch.Tensor]:
    """
    Unbatch expert weights back to list format.
    
    Args:
        batched_weights: [num_experts, out_dim, in_dim]
    
    Returns:
        List of expert weight tensors
    """
    return [batched_weights[i] for i in range(batched_weights.size(0))]


def get_optimal_block_size(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    device: torch.device
) -> int:
    """
    Determine optimal block size based on model and device characteristics.
    
    This is inspired by VLLM's dynamic block size selection.
    
    Args:
        num_tokens: Number of tokens to process
        hidden_dim: Hidden dimension size
        num_experts: Number of experts
        device: Device being used
    
    Returns:
        Optimal block size
    """
    # Base block sizes to try
    candidate_blocks = [16, 32, 64, 128, 256]
    
    # Filter blocks that are reasonable for the given dimensions
    valid_blocks = [b for b in candidate_blocks if b <= num_tokens and b <= hidden_dim]
    
    if not valid_blocks:
        # Fallback to smallest block
        return min(candidate_blocks)
    
    # Choose block size based on device capabilities
    if device.type == 'cuda':
        # For CUDA, prefer larger blocks for better utilization
        if num_tokens >= 128 and hidden_dim >= 128:
            return max(valid_blocks)
        else:
            # For smaller models, use medium blocks
            return valid_blocks[len(valid_blocks) // 2]
    else:
        # For CPU, prefer smaller blocks to avoid memory issues
        return min(valid_blocks)


def calculate_expert_load_balance(
    topk_ids: torch.Tensor,
    num_experts: int
) -> Tuple[torch.Tensor, float]:
    """
    Calculate expert load balance metrics.
    
    Args:
        topk_ids: [batch_size, top_k] expert assignments
        num_experts: Total number of experts
    
    Returns:
        expert_loads: [num_experts] load per expert
        load_balance_score: 0-1 score (higher is more balanced)
    """
    flat_ids = topk_ids.flatten()
    expert_loads = torch.bincount(flat_ids, minlength=num_experts).float()
    
    # Calculate coefficient of variation (lower is more balanced)
    mean_load = expert_loads.mean()
    std_load = expert_loads.std()
    
    if mean_load == 0:
        load_balance_score = 0.0
    else:
        cv = std_load / mean_load
        # Convert to 0-1 scale (lower CV = higher score)
        load_balance_score = max(0.0, 1.0 - cv / 2.0)
    
    return expert_loads, load_balance_score


def create_expert_mask(
    topk_ids: torch.Tensor,
    expert_id: int,
    num_experts: int = None
) -> torch.Tensor:
    """
    Create a boolean mask for tokens assigned to a specific expert.
    
    Args:
        topk_ids: [batch_size, top_k] expert assignments
        expert_id: Expert ID to create mask for
        num_experts: Total number of experts (for validation)
    
    Returns:
        Boolean mask [batch_size, top_k] indicating tokens for this expert
    """
    if num_experts is not None and expert_id >= num_experts:
        raise ValueError(f"expert_id {expert_id} exceeds num_experts {num_experts}")
    
    return topk_ids == expert_id


def optimize_expert_order(
    expert_loads: torch.Tensor,
    expert_ids: torch.Tensor
) -> torch.Tensor:
    """
    Reorder experts by load for better cache locality.
    
    Args:
        expert_loads: [num_experts] load per expert
        expert_ids: [num_blocks] expert ID for each block
    
    Returns:
        Reordered expert_ids with most loaded experts first
    """
    # Sort experts by load (descending)
    load_order = torch.argsort(expert_loads, descending=True)
    
    # Create mapping from original expert ID to load order position
    expert_position = torch.zeros_like(expert_loads, dtype=torch.long)
    expert_position[load_order] = torch.arange(len(expert_loads), dtype=torch.long, device=expert_loads.device)
    
    # Reorder expert_ids based on load position
    reordered_ids = expert_position[expert_ids]
    
    return reordered_ids


class MoEProcessingCache:
    """
    Cache for MoE processing to avoid repeated allocations.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.cache = {}
    
    def get_or_create(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        create_fn: callable = None
    ) -> torch.Tensor:
        """
        Get cached tensor or create new one.
        
        Args:
            key: Cache key
            shape: Tensor shape
            dtype: Tensor dtype
            create_fn: Optional custom creation function
        
        Returns:
            Cached or new tensor
        """
        if key in self.cache:
            cached_tensor = self.cache[key]
            if cached_tensor.shape == shape and cached_tensor.dtype == dtype:
                return cached_tensor
        
        # Create new tensor
        if create_fn is not None:
            tensor = create_fn(shape, dtype, self.device)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
        
        self.cache[key] = tensor
        return tensor
    
    def clear(self):
        """Clear all cached tensors."""
        self.cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get statistics about cache usage."""
        total_memory = 0
        tensor_count = len(self.cache)
        
        for tensor in self.cache.values():
            total_memory += tensor.numel() * tensor.element_size()
        
        return {
            'tensor_count': tensor_count,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024)
        }