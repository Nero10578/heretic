# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
VLLM Hybrid Integration for Heretic

This module provides a hybrid approach that combines VLLM's tensor parallel inference
speed with heretic's existing on-the-fly abliteration system.

The key insight is that VLLM and on-the-fly abliteration can work together:
1. Use VLLM for fast inference during refusal counting (evaluation phase)
2. Use standard heretic model with on-the-fly abliteration for actual abliteration trials
3. Optionally use VLLM with LoRA for abliteration testing (experimental)

This gives the best of both worlds: VLLM's speed for evaluation and heretic's
proven abliteration system for the actual modification.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
import torch

from .config import Settings
from .model import Model
from .vllm_inference import create_vllm_evaluator
from .utils import print


class VLLMHybridModel:
    """
    Hybrid model that combines VLLM inference speed with heretic's abliteration system.
    
    Strategy:
    - Use VLLM for fast refusal counting during evaluation
    - Use standard heretic model for abliteration trials with on-the-fly modification
    - Maintain full compatibility with existing abliteration system
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize hybrid model.
        
        Args:
            settings: Heretic settings object
        """
        self.settings = settings
        self.use_vllm_for_evaluation = getattr(settings, 'use_vllm_for_refusals', True)
        self.use_vllm_for_abliteration = getattr(settings, 'vllm_use_abliteration_for_refusals', False)
        
        # Initialize standard heretic model (for abliteration)
        print("* Initializing standard heretic model for abliteration...")
        self.heretic_model = Model(settings)
        
        # Initialize VLLM evaluator (for fast evaluation)
        self.vllm_evaluator = None
        if self.use_vllm_for_evaluation:
            print("* Initializing VLLM for fast evaluation...")
            try:
                self.vllm_evaluator = create_vllm_evaluator(settings, settings.model)
                if self.vllm_evaluator.fallback_to_standard:
                    print("* VLLM not available, using standard model for all operations")
                    self.vllm_evaluator = None
                    self.use_vllm_for_evaluation = False
                else:
                    print("* VLLM initialized successfully for evaluation")
            except Exception as e:
                print(f"* [yellow]Failed to initialize VLLM: {e}[/]")
                print("* Using standard model for all operations")
                self.vllm_evaluator = None
                self.use_vllm_for_evaluation = False
        
        # Performance tracking
        self.performance_stats = {
            'vllm_evaluations': 0,
            'standard_evaluations': 0,
            'abliteration_trials': 0,
            'vllm_time_saved': 0.0
        }
    
    def get_residuals(self, prompts: List[str]) -> torch.Tensor:
        """
        Get residuals using standard heretic model (required for abliteration).
        
        Args:
            prompts: List of input prompts
            
        Returns:
            Residual tensors
        """
        return self.heretic_model.get_residuals_batched(prompts)
    
    def get_logprobs(self, prompts: List[str]) -> torch.Tensor:
        """
        Get log probabilities using standard heretic model (required for KL divergence).
        
        Args:
            prompts: List of input prompts
            
        Returns:
            Log probability tensors
        """
        return self.heretic_model.get_logprobs_batched(prompts)
    
    def count_refusals(self, bad_prompts: List[str], use_abliteration: bool = False) -> int:
        """
        Count refusals using VLLM for speed or standard model as fallback.
        
        Args:
            bad_prompts: List of prompts that should trigger refusals
            use_abliteration: Whether to apply abliteration (if available)
            
        Returns:
            Number of refusals detected
        """
        if self.use_vllm_for_evaluation and self.vllm_evaluator is not None:
            # Use VLLM for fast evaluation
            start_time = time.perf_counter()
            
            if use_abliteration and self.use_vllm_for_abliteration:
                # Try to use VLLM with abliteration (experimental)
                try:
                    refusals = self.vllm_evaluator.count_refusals_vllm(bad_prompts, use_abliteration=True)
                    self.performance_stats['vllm_evaluations'] += 1
                    
                    elapsed = time.perf_counter() - start_time
                    self.performance_stats['vllm_time_saved'] += elapsed
                    
                    print(f"* VLLM evaluation with abliteration: {refusals}/{len(bad_prompts)} refusals")
                    return refusals
                except Exception as e:
                    print(f"* [yellow]VLLM abliteration failed: {e}[/]")
                    print("* Falling back to VLLM without abliteration")
            
            # Use VLLM without abliteration (standard evaluation)
            refusals = self.vllm_evaluator.count_refusals_vllm(bad_prompts, use_abliteration=False)
            self.performance_stats['vllm_evaluations'] += 1
            
            elapsed = time.perf_counter() - start_time
            self.performance_stats['vllm_time_saved'] += elapsed
            
            print(f"* VLLM evaluation: {refusals}/{len(bad_prompts)} refusals")
            return refusals
        
        else:
            # Use standard heretic model
            start_time = time.perf_counter()
            
            if use_abliteration:
                # Apply abliteration parameters if available
                if hasattr(self.heretic_model, 'abliteration_params') and self.heretic_model.abliteration_params:
                    print("* Using standard model with on-the-fly abliteration")
                else:
                    print("* Using standard model (no abliteration parameters)")
            
            responses = self.heretic_model.get_responses_batched(bad_prompts)
            
            # Count refusals using the same logic as evaluator
            refusals = 0
            for response in responses:
                if self._is_refusal(response):
                    refusals += 1
            
            self.performance_stats['standard_evaluations'] += 1
            
            elapsed = time.perf_counter() - start_time
            print(f"* Standard evaluation: {refusals}/{len(bad_prompts)} refusals ({elapsed:.2f}s)")
            
            return refusals
    
    def _is_refusal(self, response: str) -> bool:
        """
        Check if response is a refusal (same logic as standard evaluator).
        
        Args:
            response: Generated response
            
        Returns:
            True if response is a refusal
        """
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")
        
        # Normalize typographic apostrophes ("won't" -> "won't").
        response = response.replace("'", "'")
        
        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True
        
        return False
    
    def abliterate(self, refusal_directions: torch.Tensor, direction_index: float | None, 
                  parameters: Dict[str, Any], good_residuals: torch.Tensor = None, 
                  bad_residuals: torch.Tensor = None):
        """
        Apply abliteration using standard heretic model (proven system).
        
        Args:
            refusal_directions: Refusal direction tensors
            direction_index: Direction index
            parameters: Abliteration parameters
            good_residuals: Good residual tensors
            bad_residuals: Bad residual tensors
        """
        print("* Applying abliteration using standard heretic model...")
        self.heretic_model.abliterate(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
        self.performance_stats['abliteration_trials'] += 1
        
        # Update VLLM evaluator if it supports abliteration
        if self.vllm_evaluator and hasattr(self.vllm_evaluator, 'apply_abliteration'):
            abliteration_params = {
                'refusal_directions': refusal_directions,
                'direction_index': direction_index,
                'parameters': parameters,
                'good_residuals': good_residuals,
                'bad_residuals': bad_residuals
            }
            try:
                self.vllm_evaluator.apply_abliteration(abliteration_params)
                print("* Abliteration parameters also applied to VLLM evaluator")
            except Exception as e:
                print(f"* [yellow]Could not apply abliteration to VLLM: {e}[/]")
    
    def get_responses(self, prompts: List[str], use_abliteration: bool = True) -> List[str]:
        """
        Get responses using the appropriate model.
        
        Args:
            prompts: List of input prompts
            use_abliteration: Whether to use abliteration (if available)
            
        Returns:
            List of generated responses
        """
        if use_abliteration and hasattr(self.heretic_model, 'abliteration_params') and self.heretic_model.abliteration_params:
            # Use standard heretic model with on-the-fly abliteration
            return self.heretic_model.get_responses_batched(prompts)
        else:
            # Use VLLM for speed if available, otherwise standard model
            if self.use_vllm_for_evaluation and self.vllm_evaluator is not None:
                return self.vllm_evaluator.vllm_wrapper.get_responses_batched(prompts, use_abliteration=False)
            else:
                return self.heretic_model.get_responses_batched(prompts)
    
    def stream_chat_response(self, chat: List[Dict[str, str]]) -> str:
        """
        Stream chat response using standard heretic model (with abliteration support).
        
        Args:
            chat: Chat history
            
        Returns:
            Generated response
        """
        return self.heretic_model.stream_chat_response(chat)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the hybrid model.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self.performance_stats.copy()
        
        # Add VLLM stats if available
        if self.vllm_evaluator:
            vllm_stats = self.vllm_evaluator.get_performance_stats()
            stats['vllm_stats'] = vllm_stats
        
        # Add configuration info
        stats['configuration'] = {
            'use_vllm_for_evaluation': self.use_vllm_for_evaluation,
            'use_vllm_for_abliteration': self.use_vllm_for_abliteration,
            'vllm_available': self.vllm_evaluator is not None,
            'abliteration_available': hasattr(self.heretic_model, 'abliteration_params') and self.heretic_model.abliteration_params is not None
        }
        
        return stats
    
    def cleanup(self):
        """Clean up all models and free memory."""
        if self.vllm_evaluator:
            self.vllm_evaluator.cleanup()
        
        # Standard heretic model cleanup is handled by the model itself
        print("* Hybrid model cleaned up")


def create_hybrid_model(settings: Settings) -> VLLMHybridModel:
    """
    Factory function to create a hybrid model.
    
    Args:
        settings: Heretic settings object
        
    Returns:
        VLLMHybridModel instance
    """
    return VLLMHybridModel(settings)


# Compatibility functions for existing heretic code
def create_model_with_vllm_support(settings: Settings) -> Model:
    """
    Create a model that supports VLLM acceleration while maintaining full compatibility.
    
    This function returns a standard heretic Model instance but with VLLM evaluation
    capabilities integrated into the evaluator.
    
    Args:
        settings: Heretic settings object
        
    Returns:
        Standard heretic Model instance
    """
    # Create standard model
    model = Model(settings)
    
    # The VLLM integration happens at the evaluator level, not the model level
    # This maintains full compatibility with existing abliteration system
    
    return model