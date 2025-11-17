# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
Phase 1 optimized model class with essential MoE enhancements.
This module extends the original model.py with high-impact, low-risk optimizations.
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
from .moe_utils_phase1 import Phase1MoEOptimizer, update_matrix_in_place


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Phase1OptimizedModel(Model):
    """
    Phase 1 optimized model class with essential MoE optimizations.
    
    This class extends the original Model class with:
    - Expert batching for vectorized processing
    - Simple memory caching
    - Basic performance monitoring
    - Full backward compatibility
    """
    
    def __init__(self, settings: Settings):
        """Initialize the Phase 1 optimized model."""
        super().__init__(settings)
        
        # Initialize Phase 1 MoE optimizer
        self.moe_optimizer = Phase1MoEOptimizer()
        
        # Phase 1 optimization settings
        self.enable_phase1_optimizations = getattr(settings, 'enable_phase1_optimizations', True)
        self.phase1_batch_experts = getattr(settings, 'phase1_batch_experts', True)
        self.phase1_memory_efficient = getattr(settings, 'phase1_memory_efficient', True)
        
        print(f"* Phase 1 optimizations enabled: {self.enable_phase1_optimizations}")
        if self.enable_phase1_optimizations:
            print(f"* Expert batching: {self.phase1_batch_experts}")
            print(f"* Memory efficient mode: {self.phase1_memory_efficient}")
    
    def get_layer_matrices_phase1(self, layer_index: int) -> dict[str, list[Tensor]]:
        """
        Phase 1 optimized layer matrix processing.
        
        This method uses expert batching for MoE layers while maintaining
        full compatibility with non-MoE models.
        """
        if not self.enable_phase1_optimizations:
            # Fallback to original method
            return self.get_layer_matrices(layer_index)
        
        layer = self.get_layers()[layer_index]
        
        # Use Phase 1 optimizer for efficient processing
        matrices = self.moe_optimizer.process_layer_matrices(layer, layer_index)
        
        # Ensure we have at least one MLP down-projection
        if "mlp.down_proj" not in matrices and "mlp.down_proj.batched" not in matrices:
            # Fallback to original logic if no MoE structure detected
            with suppress(Exception):
                matrices["mlp.down_proj"] = [layer.mlp.down_proj.weight]
        
        return matrices
    
    def abliterate_phase1(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ) -> None:
        """
        Phase 1 optimized abliteration with expert batching.
        
        This method applies abliteration using the Phase 1 optimizations
        for immediate performance gains.
        """
        if not self.enable_phase1_optimizations:
            # Fallback to original method
            return self.abliterate(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
        
        print("* Using Phase 1 optimized abliteration (expert batching)... ", end="")
        
        try:
            self._apply_phase1_abliteration(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
            
            # Report performance stats
            stats = self.moe_optimizer.get_performance_stats()
            print("[green]Ok[/]")
            print(f"* Processed {stats['layers_processed']} layers")
            print(f"* Processed {stats['experts_processed']} experts in {stats['batches_processed']} batches")
            if stats['shared_experts_processed'] > 0:
                print(f"* Processed {stats['shared_experts_processed']} shared experts")
            
        except Exception as error:
            print(f"[red]Failed[/] ({error})")
            print("* Falling back to standard abliteration...")
            self.abliterate(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
    
    def _apply_phase1_abliteration(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ):
        """Apply Phase 1 abliteration using expert batching."""
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
            # Get optimized layer matrices
            layer_matrices = self.get_layer_matrices_phase1(layer_index)
            
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
                
                # Apply Phase 1 optimized abliteration
                modified_matrices = self.moe_optimizer.apply_abliteration_to_matrices(
                    {component: matrices},
                    layer_refusal_direction,
                    scale_factor * weight
                )
                
                # Update matrices in place
                for original_matrix, modified_matrix in zip(matrices, modified_matrices[component]):
                    update_matrix_in_place(original_matrix, modified_matrix)
                
                # Memory efficient cleanup
                if self.phase1_memory_efficient:
                    empty_cache()
    
    def get_residuals_batched_phase1(self, prompts: list[str]) -> Tensor:
        """
        Phase 1 optimized batched residual computation.
        
        This method implements simple batching optimizations for
        better memory efficiency.
        """
        if not self.enable_phase1_optimizations:
            return self.get_residuals_batched(prompts)
        
        # Use optimized batch size for Phase 1
        optimal_batch_size = self._get_phase1_batch_size(len(prompts))
        
        residuals = []
        for batch in batchify(prompts, optimal_batch_size):
            batch_residuals = self._compute_residuals_batch_internal(batch)
            residuals.append(batch_residuals)
            
            # Memory efficient cleanup
            if self.phase1_memory_efficient:
                empty_cache()
        
        return torch.cat(residuals, dim=0)
    
    def _get_phase1_batch_size(self, total_prompts: int) -> int:
        """Determine optimal batch size for Phase 1 optimizations."""
        # Conservative batch size for Phase 1
        max_batch_size = getattr(self.settings, 'phase1_max_batch_size', 16)
        configured_batch_size = self.settings.batch_size if self.settings.batch_size > 0 else 8
        
        # Choose the minimum for stability
        optimal_size = min(configured_batch_size, max_batch_size, total_prompts)
        
        # Ensure reasonable minimum
        if optimal_size < 1:
            optimal_size = 1
        
        return optimal_size
    
    def _compute_residuals_batch_internal(self, batch: list[str]) -> Tensor:
        """Internal batch computation (unchanged for Phase 1)."""
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
    
    def create_phase1_abliteration_hook(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ) -> 'Phase1AbliterationHook':
        """
        Create a Phase 1 optimized abliteration hook for on-the-fly processing.
        
        This returns a hook that uses expert batching for MoE models.
        """
        if self.enable_phase1_optimizations:
            return Phase1AbliterationHook(
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
    
    def get_phase1_performance_stats(self) -> dict:
        """Get Phase 1 performance statistics."""
        stats = self.moe_optimizer.get_performance_stats()
        
        # Add configuration info
        stats['configuration'] = {
            'phase1_optimizations_enabled': self.enable_phase1_optimizations,
            'batch_experts': self.phase1_batch_experts,
            'memory_efficient': self.phase1_memory_efficient,
        }
        
        return stats
    
    def reset_phase1_stats(self):
        """Reset Phase 1 performance statistics."""
        self.moe_optimizer.reset_stats()


class Phase1AbliterationHook(AbliterationHook):
    """
    Phase 1 optimized abliteration hook with expert batching.
    
    This hook integrates with the existing heretic infrastructure but
    provides essential optimizations for MoE models.
    """
    
    def __init__(
        self,
        model,
        refusal_directions: Tensor,
        direction_index: Optional[float],
        parameters: dict,
        good_residuals: Optional[Tensor] = None,
        bad_residuals: Optional[Tensor] = None,
        enable_optimizations: bool = True
    ):
        """Initialize Phase 1 abliteration hook."""
        super().__init__(model, refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
        
        self.enable_optimizations = enable_optimizations
        
        # Initialize Phase 1 optimizer if optimizations are enabled
        if enable_optimizations and hasattr(model, 'moe_optimizer'):
            self.moe_optimizer = model.moe_optimizer
        else:
            self.moe_optimizer = None
    
    def make_hook_fn(self, layer_index: int, component: str):
        """Create a Phase 1 optimized hook function."""
        if not self.enable_optimizations or self.moe_optimizer is None:
            # Fallback to original hook function
            return super().make_hook_fn(layer_index, component)
        
        def hook_fn(module, input, output):
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
            
            # Apply Phase 1 optimized abliteration
            if "batched" in component:
                return self._apply_batched_expert_abliteration(
                    output, layer_refusal_direction, weight * self.scale_factor
                )
            elif "shared" in component:
                return self._apply_shared_expert_abliteration(
                    output, layer_refusal_direction, weight * self.scale_factor
                )
            else:
                # Fallback to standard abliteration
                return super().make_hook_fn(layer_index, component)(module, input, output)
        
        return hook_fn
    
    def _get_layer_refusal_direction(self, layer_index: int) -> Tensor:
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
        output: Tensor,
        refusal_direction: Tensor,
        scale_factor: float
    ) -> Tensor:
        """Apply abliteration to batched expert outputs."""
        device = output.device
        dtype = output.dtype
        
        # Reshape for batched processing
        original_shape = output.shape
        if len(output.shape) == 3:
            output_reshaped = output.view(-1, output.shape[-1])
        else:
            output_reshaped = output
        
        # Apply optimized abliteration
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
        output: Tensor,
        refusal_direction: Tensor,
        scale_factor: float
    ) -> Tensor:
        """Apply abliteration to shared expert outputs."""
        # Similar to batched but optimized for shared experts
        return self._apply_batched_expert_abliteration(output, refusal_direction, scale_factor)