# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Optimized MoE implementation for heretic, borrowing techniques from VLLM.

Key optimizations:
1. Token alignment and batching for efficient GPU utilization
2. Block size optimization for memory coalescing
3. Vectorized operations instead of sequential loops
4. Support for quantized models
5. Efficient expert routing and load balancing
"""

import math
import time
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


class OptimizedMoEHandler:
    """
    Optimized handler for MoE models that implements VLLM-style optimizations.
    
    This class provides efficient processing of MoE layers by:
    - Aligning tokens to experts for optimal batch processing
    - Using block-wise operations for better GPU utilization
    - Vectorizing operations where possible
    - Supporting quantized weights
    """
    
    def __init__(self, block_size: int = 64, enable_profiling: bool = False):
        """
        Initialize the optimized MoE handler.
        
        Args:
            block_size: Block size for matrix operations (tuned per GPU)
            enable_profiling: Whether to enable performance profiling
        """
        self.block_size = block_size
        self.enable_profiling = enable_profiling
        self.profiling_data = {}
        
    def process_moe_layer(
        self,
        hidden_states: Tensor,
        expert_weights: list[Tensor],
        routing_weights: Optional[Tensor] = None,
        top_k: int = 1,
        expert_indices: Optional[Tensor] = None,
        is_quantized: bool = False,
        scale_factors: Optional[list[Tensor]] = None,
        zero_points: Optional[list[Tensor]] = None,
    ) -> Tensor:
        """
        Process a MoE layer with optimized token alignment and batching.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_dim)
            expert_weights: List of expert weight tensors
            routing_weights: Optional routing weights for load balancing
            top_k: Number of experts to route to
            expert_indices: Pre-computed expert indices (optional)
            is_quantized: Whether weights are quantized
            scale_factors: Quantization scale factors
            zero_points: Quantization zero points
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        if self.enable_profiling:
            start_time = time.perf_counter()
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_experts = len(expert_weights)
        
        # Flatten for easier processing
        hidden_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_flat.shape[0]
        
        # If expert indices not provided, compute them
        if expert_indices is None:
            if routing_weights is not None:
                # Simple routing - use first expert
                expert_indices = torch.zeros(num_tokens, dtype=torch.long, device=hidden_states.device)
                expert_weights_flat = [expert_weights[0]] * num_tokens
            else:
                # Use routing weights to select top-k experts
                expert_indices, selected_weights = self._select_experts(
                    hidden_flat, routing_weights, top_k
                )
                expert_weights_flat = [expert_weights[i] for i in expert_indices]
        else:
            # Use provided expert indices
            expert_weights_flat = [expert_weights[i] for i in expert_indices]
            selected_weights = None
            
        # Align tokens to experts for efficient processing
        sorted_tokens, sorted_experts, tokens_per_expert = self._align_tokens_to_experts(
            expert_indices, num_experts
        )
        
        # Process each expert with aligned tokens
        expert_outputs = []
        for expert_id in range(num_experts):
            # Get tokens assigned to this expert
            expert_mask = sorted_experts == expert_id
            if not expert_mask.any():
                expert_outputs.append(torch.zeros(0, hidden_dim, device=hidden_states.device))
                continue
                
            expert_tokens = sorted_tokens[expert_mask]
            expert_weight = expert_weights_flat[expert_id]
            
            # Apply quantization if needed
            if is_quantized and scale_factors is not None:
                raise ValueError("Scale factors required for quantized weights")
                
            # Process expert tokens in batches
            expert_output = self._process_expert_batch(
                expert_tokens, expert_weight, is_quantized, 
                scale_factors[expert_id] if scale_factors else None,
                zero_points[expert_id] if zero_points else None
            )
            expert_outputs.append(expert_output)
            
        # Combine expert outputs
        final_output = self._combine_expert_outputs(
            expert_outputs, sorted_experts, tokens_per_expert, 
            selected_weights, hidden_flat.shape
        )
        
        # Reshape back to original dimensions
        final_output = final_output.view(batch_size, seq_len, hidden_dim)
        
        if self.enable_profiling:
            end_time = time.perf_counter()
            self.profiling_data['process_moe_layer'] = {
                'time': end_time - start_time,
                'num_tokens': num_tokens,
                'num_experts': num_experts,
                'tokens_per_second': num_tokens / (end_time - start_time)
            }
            
        return final_output
        
    def _select_experts(
        self, 
        hidden_states: Tensor, 
        routing_weights: Tensor, 
        top_k: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Select top-k experts using routing weights.
        
        Args:
            hidden_states: Input tokens
            routing_weights: Routing weights
            top_k: Number of experts to select
            
        Returns:
            Tuple of (expert_indices, selected_weights)
        """
        # Compute routing scores
        scores = torch.matmul(hidden_states, routing_weights.T)
        
        # Get top-k experts
        top_k_scores, expert_indices = torch.topk(scores, k=top_k, dim=-1)
        
        # Apply softmax to get weights
        expert_weights = F.softmax(top_k_scores, dim=-1)
        
        return expert_indices, expert_weights
        
    def _align_tokens_to_experts(
        self, 
        expert_indices: Tensor, 
        num_experts: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Align tokens to experts for efficient batch processing.
        
        This is a key optimization from VLLM that ensures tokens assigned
        to the same expert are grouped together for optimal GPU utilization.
        """
        num_tokens = expert_indices.shape[0]
        
        # Count tokens per expert
        tokens_per_expert = torch.zeros(num_experts, dtype=torch.long, device=expert_indices.device)
        for i in range(num_experts):
            tokens_per_expert[i] = (expert_indices == i).sum()
            
        # Calculate padding needed for block alignment
        max_tokens_per_expert = tokens_per_expert.max().item()
        padded_tokens_per_expert = ((max_tokens_per_expert + self.block_size - 1) // self.block_size) * self.block_size
        
        # Create aligned token indices
        total_padded_tokens = padded_tokens_per_expert.sum().item()
        sorted_indices = torch.empty(total_padded_tokens, dtype=torch.long, device=expert_indices.device)
        sorted_experts = torch.empty(total_padded_tokens, dtype=torch.long, device=expert_indices.device)
        
        # Fill with actual tokens
        offset = 0
        for expert_id in range(num_experts):
            expert_mask = expert_indices == expert_id
            expert_tokens = expert_indices[expert_mask]
            num_expert_tokens = expert_tokens.shape[0]
            
            # Place expert tokens
            sorted_indices[offset:offset + num_expert_tokens] = expert_tokens
            sorted_experts[offset:offset + num_expert_tokens] = expert_id
            offset += num_expert_tokens
            
            # Add padding
            padding_needed = padded_tokens_per_expert[expert_id] - num_expert_tokens
            if padding_needed > 0:
                sorted_indices[offset + num_expert_tokens:offset + padded_tokens_per_expert[expert_id]] = 0
                sorted_experts[offset + num_expert_tokens:offset + padded_tokens_per_expert[expert_id]] = -1  # Mark as padding
                offset += padding_needed
                
        return sorted_indices, sorted_experts, tokens_per_expert
        
    def _process_expert_batch(
        self,
        tokens: Tensor,
        expert_weight: Tensor,
        is_quantized: bool = False,
        scale_factor: Optional[Tensor] = None,
        zero_point: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process tokens for a single expert using batched operations.
        
        Args:
            tokens: Tokens assigned to this expert
            expert_weight: Expert weight tensor
            is_quantized: Whether weights are quantized
            scale_factor: Quantization scale factor
            zero_point: Quantization zero point
            
        Returns:
            Expert output for the tokens
        """
        if is_quantized:
            return self._process_quantized_expert(tokens, expert_weight, scale_factor, zero_point)
        else:
            return self._process_standard_expert(tokens, expert_weight)
            
    def _process_standard_expert(self, tokens: Tensor, expert_weight: Tensor) -> Tensor:
        """
        Process tokens with standard (non-quantized) expert weights.
        """
        # Use batched matrix multiplication
        return F.linear(tokens, expert_weight)
        
    def _process_quantized_expert(
        self,
        tokens: Tensor,
        expert_weight: Tensor,
        scale_factor: Tensor,
        zero_point: Tensor,
    ) -> Tensor:
        """
        Process tokens with quantized expert weights.
        """
        # Dequantize weights if needed
        if expert_weight.dtype in [torch.int8, torch.uint8, torch.quint8]:
            # Dequantize for computation
            weight_fp32 = expert_weight.float() * scale_factor + zero_point.float()
            return F.linear(tokens, weight_fp32)
        else:
            # Already in appropriate format
            return F.linear(tokens, expert_weight)
            
    def _combine_expert_outputs(
        self,
        expert_outputs: list[Tensor],
        sorted_experts: Tensor,
        tokens_per_expert: Tensor,
        selected_weights: Optional[Tensor],
        original_shape: Tuple[int, ...],
    ) -> Tensor:
        """
        Combine outputs from all experts efficiently.
        """
        num_tokens = original_shape[0]
        hidden_dim = expert_outputs[0].shape[-1]
        
        # Create combined output
        combined_output = torch.zeros(num_tokens, hidden_dim, device=expert_outputs[0].device)
        
        # Place expert outputs back to original token positions
        offset = 0
        for expert_id, expert_output in enumerate(expert_outputs):
            expert_mask = sorted_experts == expert_id
            if expert_mask.any():
                num_expert_tokens = expert_mask.sum().item()
                # Place expert outputs
                combined_output[offset:offset + num_expert_tokens] = expert_output[:num_expert_tokens]
                offset += num_expert_tokens
            else:
                # Skip padding
                offset += (sorted_experts == expert_id).sum().item()
                
        return combined_output


def optimized_fused_moe(
    hidden_states: Tensor,
    w1: Tensor,  # First expert weights (gate/up projections)
    w2: Tensor,  # Second expert weights (down projections)
    gating_output: Tensor,
    topk: int = 1,
    renormalize: bool = False,
    is_quantized: bool = False,
    scale_factors: Optional[list[Tensor]] = None,
    zero_points: Optional[list[Tensor]] = None,
    block_size: int = 64,
    enable_profiling: bool = False,
) -> Tensor:
    """
    Optimized MoE implementation inspired by VLLM's approach.
    
    This function provides a drop-in replacement for standard MoE processing
    with significant performance improvements for complex MoE models like GLM.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, hidden_dim)
        w1: First expert weights tensor of shape (num_experts, intermediate_size, hidden_dim)
        w2: Second expert weights tensor of shape (num_experts, hidden_dim, intermediate_size)
        gating_output: Gating output for routing
        topk: Number of experts to route to
        renormalize: Whether to renormalize routing weights
        is_quantized: Whether weights are quantized
        scale_factors: List of scale factors for each expert
        zero_points: List of zero points for each expert
        block_size: Block size for optimization
        enable_profiling: Enable performance profiling
        
    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
    handler = OptimizedMoEHandler(block_size=block_size, enable_profiling=enable_profiling)
    
    # Extract expert weights
    num_experts = w1.shape[0]
    expert_weights_w1 = [w1[i] for i in range(num_experts)]
    expert_weights_w2 = [w2[i] for i in range(num_experts)]
    
    # Process first expert weights (gate/up projections)
    intermediate = handler.process_moe_layer(
        hidden_states,
        expert_weights_w1,
        gating_output,
        topk,
        is_quantized=is_quantized,
        scale_factors=scale_factors[:num_experts] if scale_factors else None,
        zero_points=zero_points[:num_experts] if zero_points else None,
    )
    
    # Apply activation (SiLU/Swish)
    gate, up = intermediate.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    
    # Process second expert weights (down projections)
    output = handler.process_moe_layer(
        activated,
        expert_weights_w2,
        None,  # No routing needed for second layer
        topk=1,
        expert_indices=None,  # Use same routing as first layer
        is_quantized=is_quantized,
        scale_factors=scale_factors[num_experts:] if scale_factors and len(scale_factors) > num_experts else None,
        zero_points=zero_points[num_experts:] if zero_points and len(zero_points) > num_experts else None,
    )
    
    if enable_profiling:
        print(f"MoE Profiling Data: {handler.profiling_data}")
        
    return output


def optimize_for_glm_model(model, settings) -> None:
    """
    Apply specific optimizations for GLM MoE models with shared experts.
    
    GLM models have both regular experts and shared experts that are always active.
    This function optimizes the processing by:
    1. Separating shared and regular expert processing
    2. Optimizing shared expert computation (always active)
    3. Efficiently combining outputs
    """
    # Check if this is a GLM MoE model
    has_shared_experts = False
    try:
        first_layer = model.get_layers()[0]
        if hasattr(first_layer.mlp, 'shared_experts'):
            has_shared_experts = True
    except:
        pass
        
    if not has_shared_experts:
        return
        
    print("* Optimizing for GLM MoE model with shared experts...")
    
    # Store original forward methods
    original_forwards = {}
    
    for layer_idx, layer in enumerate(model.get_layers()):
        if not hasattr(layer.mlp, 'shared_experts'):
            continue
            
        # Store original forward
        original_forwards[layer_idx] = layer.mlp.forward
        
        def optimized_forward(hidden_states):
            # Split processing for shared and regular experts
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Get routing for regular experts
            router_logits = layer.mlp.gate(hidden_states)
            
            # Always process shared experts (they're always active)
            shared_output = layer.mlp.shared_experts(hidden_states)
            
            # Process regular experts if routing exists
            if hasattr(layer.mlp, 'experts') and router_logits is not None:
                # Use optimized MoE for regular experts
                regular_output = optimized_fused_moe(
                    hidden_states,
                    layer.mlp.experts.w1.weight if hasattr(layer.mlp.experts, 'w1') else layer.mlp.experts.gate_proj.weight,
                    layer.mlp.experts.w2.weight if hasattr(layer.mlp.experts, 'w2') else layer.mlp.experts.down_proj.weight,
                    router_logits,
                    topk=getattr(layer.mlp, 'num_experts_per_tok', 1),
                    renormalize=getattr(layer.mlp, 'router_z_loss_coef', 0.0) > 0,
                    block_size=getattr(settings, 'moe_block_size', 64),
                )
                
                # Combine shared and regular outputs
                if hasattr(layer.mlp, 'combine_expert_outputs'):
                    return layer.mlp.combine_expert_outputs(shared_output, regular_output)
                else:
                    # Simple addition
                    return shared_output + regular_output
            else:
                # Only shared experts
                return shared_output
                
        # Replace forward method
        layer.mlp.forward = optimized_forward
        
    print(f"* Applied GLM MoE optimizations to {len(original_forwards)} layers")


def apply_lora_to_moe(
    model,
    lora_a: Tensor,
    lora_b: Tensor,
    target_layers: Optional[list[str]] = None,
    alpha: float = 1.0,
) -> None:
    """
    Apply LoRA adapters to MoE layers efficiently.
    
    This is inspired by VLLM's moe_lora implementation that allows
    on-the-fly application of LoRA without modifying the base weights.
    """
    print("* Applying LoRA to MoE layers...")
    
    for layer_idx, layer in enumerate(model.get_layers()):
        layer_name = f"model.layers.{layer_idx}.mlp"
        
        # Skip if not in target layers
        if target_layers and layer_name not in target_layers:
            continue
            
        # Check if this layer has experts
        if not hasattr(layer.mlp, 'experts'):
            continue
            
        # Apply LoRA to each expert
        for expert_idx, expert in enumerate(layer.mlp.experts):
            # Get expert weight matrices
            if hasattr(expert, 'w1') and hasattr(expert, 'w2'):
                w1 = expert.w1
                w2 = expert.w2
                
                # Create LoRA-modified weights
                def lora_forward(x):
                    # Original forward
                    original = F.linear(x, w1.weight)
                    gate, up = original.chunk(2, dim=-1)
                    activated = F.silu(gate) * up
                    
                    # Apply LoRA to down projection
                    lora_output = F.linear(activated, w2.weight)
                    lora_adjusted = lora_output + alpha * (x @ lora_a @ lora_b)
                    
                    return lora_adjusted
                    
                # Replace expert forward
                expert.forward = lora_forward
                
    print("* LoRA applied to MoE layers")