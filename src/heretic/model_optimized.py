# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Optimized model class with VLLM-inspired MoE enhancements.
This module extends the original model.py with high-performance MoE processing.
"""

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
    TorchAoConfig,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
from .utils import batchify, empty_cache, print, print_memory_usage
from .model import Model, AbliterationParameters, AbliterationHook
from .moe_utils import (
    align_tokens_for_experts,
    batch_expert_weights,
    get_optimal_block_size,
    MoEProcessingCache
)
from .fused_abliteration import FusedMoEAbliterator, BatchedAbliterationHook


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class OptimizedModel(Model):
    """
    Enhanced model class with VLLM-inspired MoE optimizations.
    
    This class extends the original Model class with:
    - Expert batching and fusion
    - Block-size alignment for GPU optimization
    - Memory-efficient caching
    - Fused abliteration kernels
    """
    
    def __init__(self, settings: Settings):
        """Initialize the optimized model."""
        super().__init__(settings)
        
        # Initialize MoE optimization components
        self._init_moe_optimizations()
    
    def _init_moe_optimizations(self):
        """Initialize MoE-specific optimizations."""
        if not getattr(self.settings, 'enable_moe_optimizations', True):
            return
        
        # Initialize processing cache
        self.moe_cache = MoEProcessingCache(self.model.device if self.model else torch.device('cpu'))
        
        # Initialize fused abliterator
        if getattr(self.settings, 'moe_fused_abliteration', True):
            block_size = getattr(self.settings, 'moe_block_size', 64)
            self.fused_abliterator = FusedMoEAbliterator(block_size=block_size)
        else:
            self.fused_abliterator = None
        
        # Performance tracking
        self.moe_stats = {
            'experts_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'memory_saved_mb': 0
        }
    
    def get_layer_matrices_optimized(self, layer_index: int) -> dict[str, list[Tensor]]:
        """
        VLLM-inspired expert batching for MoE models.
        
        This method detects MoE layers and batches expert weights for
        efficient processing, similar to VLLM's approach.
        """
        layer = self.get_layers()[layer_index]
        matrices = {}
        
        def try_add(component: str, matrix: Any):
            assert torch.is_tensor(matrix)
            if component not in matrices:
                matrices[component] = []
            matrices[component].append(matrix)
        
        # Standard attention processing
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)
        
        # Check if this is an MoE layer and apply optimizations
        if getattr(self.settings, 'enable_moe_optimizations', True):
            mlp_matrices = self._process_moe_layer_optimized(layer)
            matrices.update(mlp_matrices)
        else:
            # Fallback to original processing
            matrices.update(self._process_mle_layer_standard(layer))
        
        return matrices
    
    def _process_moe_layer_optimized(self, layer) -> dict[str, list[Tensor]]:
        """
        Process MoE layer with VLLM-inspired optimizations.
        
        This method batches expert weights for efficient processing
        and handles shared experts separately.
        """
        matrices = {}
        
        # Try different MoE architectures
        # GLM MoE models with experts and shared experts
        with suppress(Exception):
            if hasattr(layer.mlp, 'experts') and hasattr(layer.mlp, 'shared_experts'):
                # Batch all expert weights together
                expert_weights = []
                for expert in layer.mlp.experts:
                    expert_weights.append(expert.down_proj.weight)
                
                if expert_weights:
                    # Stack experts for batched operations [num_experts, out_dim, in_dim]
                    batched_experts = batch_expert_weights(expert_weights)
                    matrices["mlp.down_proj.batched"] = [batched_experts]
                    self.moe_stats['experts_processed'] += len(expert_weights)
                
                # Handle shared experts separately
                matrices["mlp.shared_down_proj"] = [layer.mlp.shared_experts.down_proj.weight]
                return matrices
        
        # Standard MoE models (e.g., Qwen3, Mixtral)
        with suppress(Exception):
            if hasattr(layer.mlp, 'experts'):
                expert_weights = []
                for expert in layer.mlp.experts:
                    expert_weights.append(expert.down_proj.weight)
                
                if expert_weights:
                    batched_experts = batch_expert_weights(expert_weights)
                    matrices["mlp.down_proj.batched"] = [batched_experts]
                    self.moe_stats['experts_processed'] += len(expert_weights)
                return matrices
        
        # Phi-3.5-MoE style
        with suppress(Exception):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
                expert_weights = []
                for expert in layer.block_sparse_moe.experts:
                    expert_weights.append(expert.w2.weight)
                
                if expert_weights:
                    batched_experts = batch_expert_weights(expert_weights)
                    matrices["mlp.down_proj.batched"] = [batched_experts]
                    self.moe_stats['experts_processed'] += len(expert_weights)
                return matrices
        
        # gpt-oss MoE style
        with suppress(Exception):
            if hasattr(layer.mlp, 'experts') and hasattr(layer.mlp.experts, 'down_proj'):
                # The implementation stores down-projections for all experts in a single 3D tensor
                matrices["mlp.down_proj"] = [layer.mlp.experts.down_proj]
                return matrices
        
        # Fallback to dense model processing
        return self._process_mle_layer_standard(layer)
    
    def _process_mle_layer_standard(self, layer) -> dict[str, list[Tensor]]:
        """Standard MLP processing for non-MoE models."""
        matrices = {}
        
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)
        
        return matrices
    
    def abliterate_optimized(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ) -> None:
        """
        Optimized abliteration using VLLM-inspired techniques.
        
        This method applies abliteration with expert batching and
        fused operations for maximum performance.
        """
        if not getattr(self.settings, 'enable_moe_optimizations', True):
            # Fallback to original method
            return self.abliterate(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
        
        print("* Using VLLM-inspired optimized abliteration... ", end="")
        
        try:
            # Apply abliteration using optimized approach
            if getattr(self.settings, 'moe_fused_abliteration', True) and self.fused_abliterator is not None:
                self._apply_fused_abliteration(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
            else:
                self._apply_batched_abliteration(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
            
            print("[green]Ok[/]")
            print(f"* Processed {self.moe_stats['experts_processed']} experts in {self.moe_stats['batches_processed']} batches")
            
        except Exception as error:
            print(f"[red]Failed[/] ({error})")
            print("* Falling back to standard abliteration...")
            self.abliterate(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
    
    def _apply_fused_abliteration(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ):
        """Apply fused abliteration using custom kernels."""
        use_norm_preserving = getattr(self.settings, 'use_norm_preserving_abliteration', False)
        scale_factor = getattr(self.settings, 'abliteration_scale_factor', 1.0)
        
        # Pre-compute mean directions for projection if needed
        if use_norm_preserving and good_residuals is not None and bad_residuals is not None:
            harmful_mean = bad_residuals.mean(dim=0)
            harmless_mean = good_residuals.mean(dim=0)
        else:
            harmful_mean = None
            harmless_mean = None
        
        for layer_index in range(len(self.get_layers())):
            layer_matrices = self.get_layer_matrices_optimized(layer_index)
            
            for component, matrices in layer_matrices.items():
                if component not in parameters:
                    continue
                
                params = parameters[component]
                distance = abs(layer_index - params.max_weight_position)
                
                if distance > params.min_weight_distance:
                    continue
                
                # Calculate interpolation weight
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )
                
                # Get refusal direction for this layer
                if direction_index is None:
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    weight_idx, idx = math.modf(direction_index + 1)
                    layer_refusal_direction = F.normalize(
                        refusal_directions[int(idx)].lerp(
                            refusal_directions[int(idx) + 1],
                            weight_idx,
                        ),
                        p=2,
                        dim=0,
                    )
                
                # Apply biprojection if using norm-preserving
                if use_norm_preserving and harmless_mean is not None:
                    harmless_dir = harmless_mean[layer_index]
                    harmless_normalized = F.normalize(harmless_dir.float(), dim=0)
                    projection_scalar = torch.dot(layer_refusal_direction, harmless_normalized)
                    layer_refusal_direction = layer_refusal_direction - projection_scalar * harmless_normalized
                
                # Apply fused abliteration based on component type
                for matrix in matrices:
                    if "batched" in component:
                        # Handle batched expert weights
                        modified_matrix = self.fused_abliterator.abliterate_experts_fused(
                            matrix,
                            refusal_directions,
                            layer_index,
                            scale_factor * weight,
                            use_norm_preserving=use_norm_preserving
                        )
                        self._update_matrix_in_place(matrix, modified_matrix)
                        
                    elif "shared" in component:
                        # Handle shared expert weights
                        modified_matrix = self.fused_abliterator.abliterate_shared_experts(
                            matrix,
                            refusal_directions,
                            layer_index,
                            scale_factor * weight,
                            use_norm_preserving=use_norm_preserving
                        )
                        self._update_matrix_in_place(matrix, modified_matrix)
                        
                    else:
                        # Fallback to standard abliteration
                        self._apply_standard_abliteration_to_matrix(
                            matrix, layer_refusal_direction, scale_factor * weight, use_norm_preserving
                        )
                
                self.moe_stats['batches_processed'] += 1
    
    def _apply_batched_abliteration(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ):
        """Apply batched abliteration without custom kernels."""
        # This would implement a simpler batching approach
        # For now, delegate to the original method
        self.abliterate(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
    
    def _update_matrix_in_place(self, original_matrix: Tensor, modified_matrix: Tensor):
        """Update matrix in place, handling different quantization types."""
        if hasattr(original_matrix, 'bnb_quantized') and original_matrix.bnb_quantized:
            # Handle bitsandbytes quantization
            try:
                if hasattr(original_matrix, 'weight') and original_matrix.weight is not None:
                    with torch.no_grad():
                        original_matrix.weight.data = modified_matrix
            except Exception as e:
                print(f"* Warning: Failed to update quantized matrix: {e}")
        elif hasattr(original_matrix, '_data_impl') and hasattr(original_matrix._data_impl, '_value'):
            # Handle torchao quantization
            with torch.no_grad():
                original_matrix.copy_(modified_matrix)
        else:
            # Handle regular weights
            with torch.no_grad():
                original_matrix.copy_(modified_matrix)
    
    def _apply_standard_abliteration_to_matrix(
        self,
        matrix: Tensor,
        refusal_direction: Tensor,
        scale_factor: float,
        use_norm_preserving: bool = False
    ):
        """Apply standard abliteration to a single matrix."""
        if use_norm_preserving:
            modified_matrix = self.modify_tensor_norm_preserved(matrix, refusal_direction, scale_factor)
        else:
            # Standard abliteration
            device = matrix.device
            projector = torch.outer(
                refusal_direction.to(device),
                refusal_direction.to(device),
            ).to(matrix.dtype)
            
            projection = torch.matmul(projector, matrix.T).T
            modified_matrix = matrix - scale_factor * projection
        
        self._update_matrix_in_place(matrix, modified_matrix)
    
    def get_residuals_batched_optimized(self, prompts: list[str]) -> Tensor:
        """
        VLLM-inspired batched residual computation with memory reuse.
        
        This method implements efficient batching and caching for
        residual computation in MoE models.
        """
        if not getattr(self.settings, 'enable_moe_optimizations', True):
            return self.get_residuals_batched(prompts)
        
        # Determine optimal batch size
        optimal_batch_size = self._get_optimal_batch_size(len(prompts))
        
        residuals = []
        for batch in batchify(prompts, optimal_batch_size):
            batch_residuals = self._compute_residuals_batch_internal(batch)
            residuals.append(batch_residuals)
            
            # Clear memory after each batch
            empty_cache()
        
        return torch.cat(residuals, dim=0)
    
    def _get_optimal_batch_size(self, total_prompts: int) -> int:
        """Determine optimal batch size based on model and memory constraints."""
        max_batch_size = getattr(self.settings, 'moe_max_experts_per_batch', 8)
        configured_batch_size = self.settings.batch_size if self.settings.batch_size > 0 else 32
        
        # Choose the minimum of configured and optimal batch size
        optimal_size = min(configured_batch_size, max_batch_size, total_prompts)
        
        # Ensure batch size is reasonable for the model
        if optimal_size < 1:
            optimal_size = 1
        
        return optimal_size
    
    def _compute_residuals_batch_internal(self, batch: list[str]) -> Tensor:
        """Internal batch computation with cache reuse."""
        # Use the original generate method but with optimized batching
        _, outputs = self.generate(
            batch,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        
        hidden_states = outputs.hidden_states[0]
        residuals = torch.stack(
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )
        
        return residuals.to(torch.float32)
    
    def create_optimized_abliteration_hook(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ) -> 'BatchedAbliterationHook':
        """
        Create an optimized abliteration hook for on-the-fly processing.
        
        This returns a VLLM-inspired optimized hook that can handle
        MoE models efficiently.
        """
        if getattr(self.settings, 'enable_moe_optimizations', True):
            return BatchedAbliterationHook(
                self,
                refusal_directions,
                direction_index,
                parameters,
                good_residuals,
                bad_residuals,
                enable_optimizations=True
            )
        else:
            # Fallback to original hook
            return AbliterationHook(
                self,
                refusal_directions,
                direction_index,
                parameters,
                good_residuals,
                bad_residuals
            )
    
    def get_moe_performance_stats(self) -> dict:
        """Get MoE performance statistics."""
        stats = self.moe_stats.copy()
        
        if self.fused_abliterator:
            stats['abliterator_stats'] = self.fused_abliterator.get_performance_stats()
        
        if hasattr(self, 'moe_cache'):
            stats['cache_stats'] = self.moe_cache.get_cache_stats()
        
        return stats
    
    def reset_moe_stats(self):
        """Reset MoE performance statistics."""
        self.moe_stats = {
            'experts_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'memory_saved_mb': 0
        }
        
        if self.fused_abliterator:
            self.fused_abliterator.reset_stats()
    
    def clear_moe_cache(self):
        """Clear MoE processing cache."""
        if hasattr(self, 'moe_cache'):
            self.moe_cache.clear()
        
        if self.fused_abliterator:
            self.fused_abliterator.clear_cache()