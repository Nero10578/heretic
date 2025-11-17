# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
VLLM-based inference wrapper for optimized refusal counting and evaluation.
This module provides VLLM tensor parallel inference capabilities to dramatically
speed up the "Counting model refusals..." part of the heretic pipeline.

Enhanced with on-the-fly abliteration support through VLLM LoRA capabilities.
"""

import os
import time
import tempfile
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Install with: pip install vllm")

from .config import Settings
from .utils import print, empty_cache


class VLLMInferenceWrapper:
    """
    VLLM-based inference wrapper for high-performance tensor parallel inference.
    Optimized specifically for refusal counting and evaluation tasks.
    
    Enhanced with on-the-fly abliteration support through LoRA adapters.
    """
    
    def __init__(self, settings: Settings, model_path: str = None):
        """
        Initialize VLLM inference wrapper.
        
        Args:
            settings: Heretic settings object
            model_path: Path to model (defaults to settings.model)
        """
        self.settings = settings
        self.model_path = model_path or settings.model
        self.tokenizer = None
        self.vllm_engine = None
        self.is_initialized = False
        
        # VLLM configuration
        self.enable_vllm = getattr(settings, 'enable_vllm_inference', True)
        self.vllm_tensor_parallel_size = getattr(settings, 'vllm_tensor_parallel_size', 2)
        self.vllm_gpu_memory_utilization = getattr(settings, 'vllm_gpu_memory_utilization', 0.85)
        self.vllm_max_model_len = getattr(settings, 'vllm_max_model_len', 8192)
        self.vllm_batch_size = getattr(settings, 'vllm_batch_size', 32)
        self.vllm_dtype = getattr(settings, 'vllm_dtype', 'auto')
        
        # On-the-fly abliteration support
        self.enable_lora_abliteration = getattr(settings, 'vllm_enable_lora_abliteration', False)
        self.current_lora_path = None
        self.lora_request = None
        self.abliteration_cache_dir = tempfile.mkdtemp(prefix="heretic_vllm_lora_")
        
        # Performance tracking
        self.inference_stats = {
            'total_requests': 0,
            'total_time': 0.0,
            'tokens_per_second': 0.0,
            'batches_processed': 0,
            'lora_switches': 0,
            'abliteration_active': False
        }
        
    def initialize(self) -> bool:
        """
        Initialize VLLM engine and tokenizer.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not VLLM_AVAILABLE:
            print("* VLLM not available, falling back to standard inference")
            return False
            
        if not self.enable_vllm:
            print("* VLLM inference disabled in settings")
            return False
            
        try:
            print("* Initializing VLLM inference engine...")
            print(f"  * Model: {self.model_path}")
            print(f"  * Tensor parallel size: {self.vllm_tensor_parallel_size}")
            print(f"  * GPU memory utilization: {self.vllm_gpu_memory_utilization}")
            print(f"  * Max model length: {self.vllm_max_model_len}")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
            
            # Initialize VLLM engine with optimized settings for refusal counting
            # Enable LoRA support if on-the-fly abliteration is requested
            enable_lora = self.enable_lora_abliteration
            
            self.vllm_engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_model_len=self.vllm_max_model_len,
                dtype=self.vllm_dtype,
                # Optimized settings for short text generation (refusal detection)
                max_num_batched_tokens=self.vllm_batch_size * 100,  # Rough estimate
                max_num_seqs=self.vllm_batch_size,
                # Disable KV cache for short sequences to save memory
                enable_chunked_prefill=False,
                # Optimize for throughput
                use_v2_block_manager=True,
                swap_space=4,  # 4GB swap space
                # LoRA support for on-the-fly abliteration
                enable_lora=enable_lora,
                max_loras=1 if enable_lora else 0,
                lora_extra_vocab_size=0,
                lora_dtype=self.vllm_dtype,
            )
            
            if enable_lora:
                print("* VLLM LoRA support enabled for on-the-fly abliteration")
            
            self.is_initialized = True
            print("* VLLM inference engine initialized successfully")
            return True
            
        except Exception as e:
            print(f"* [red]Failed to initialize VLLM: {e}[/]")
            print("* Falling back to standard inference")
            self.is_initialized = False
            return False
    
    def get_chat_format(self, prompt: str) -> str:
        """
        Format prompt using chat template.
        
        Args:
            prompt: Raw prompt string
            
        Returns:
            Formatted chat prompt
        """
        # Check if tokenizer supports system role
        try:
            test_chat = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
            ]
            self.tokenizer.apply_chat_template(test_chat, tokenize=False)
            # If no exception, system role is supported
            chat = [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": prompt},
            ]
        except Exception:
            # If system role is not supported, prepend it to the user message
            modified_prompt = f"{self.settings.system_prompt}\n\n{prompt}"
            chat = [{"role": "user", "content": modified_prompt}]
        
        return self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
    
    def generate_responses(self, prompts: List[str], max_tokens: int = 50, use_lora: bool = None) -> List[str]:
        """
        Generate responses using VLLM with optimized batching.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate (default 50 for refusal detection)
            use_lora: Whether to use LoRA abliteration (None = auto-detect)
            
        Returns:
            List of generated responses
        """
        if not self.is_initialized:
            raise RuntimeError("VLLM engine not initialized. Call initialize() first.")
        
        # Determine if we should use LoRA
        if use_lora is None:
            use_lora = self.lora_request is not None
        
        # Format prompts with chat template
        formatted_prompts = [self.get_chat_format(prompt) for prompt in prompts]
        
        # Configure sampling parameters for refusal detection
        # Use greedy decoding for deterministic results
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding
            top_p=1.0,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True,
        )
        
        # Process in batches to optimize throughput
        all_responses = []
        start_time = time.perf_counter()
        
        for i in range(0, len(formatted_prompts), self.vllm_batch_size):
            batch_prompts = formatted_prompts[i:i + self.vllm_batch_size]
            
            # Generate responses for this batch
            if use_lora and self.lora_request:
                outputs = self.vllm_engine.generate(
                    batch_prompts,
                    sampling_params,
                    lora_request=self.lora_request
                )
            else:
                outputs = self.vllm_engine.generate(batch_prompts, sampling_params)
            
            # Extract generated text
            batch_responses = []
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                batch_responses.append(generated_text)
            
            all_responses.extend(batch_responses)
            self.inference_stats['batches_processed'] += 1
        
        # Update performance stats
        total_time = time.perf_counter() - start_time
        self.inference_stats['total_requests'] += len(prompts)
        self.inference_stats['total_time'] += total_time
        
        if total_time > 0:
            self.inference_stats['tokens_per_second'] = (
                self.inference_stats['total_requests'] * max_tokens / total_time
            )
        
        return all_responses
    
    def get_responses_batched(self, prompts: List[str], use_abliteration: bool = None) -> List[str]:
        """
        Get responses with automatic batching (compatible with heretic's interface).
        
        Args:
            prompts: List of input prompts
            use_abliteration: Whether to use abliteration (None = auto-detect)
            
        Returns:
            List of generated responses
        """
        if not self.is_initialized:
            raise RuntimeError("VLLM engine not initialized. Call initialize() first.")
        
        return self.generate_responses(prompts, max_tokens=50, use_lora=use_abliteration)
    
    def apply_abliteration_lora(self, abliteration_params: Dict[str, Any]) -> bool:
        """
        Apply abliteration as a LoRA adapter for on-the-fly modification.
        
        Args:
            abliteration_params: Abliteration parameters including directions and weights
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_lora_abliteration:
            print("* LoRA abliteration not enabled in VLLM configuration")
            return False
        
        if not self.is_initialized:
            print("* VLLM engine not initialized")
            return False
        
        try:
            # Create a temporary LoRA adapter with abliteration weights
            lora_path = self._create_abliteration_lora(abliteration_params)
            
            if lora_path:
                # Remove existing LoRA if any
                if self.lora_request:
                    self._remove_current_lora()
                
                # Apply new LoRA
                self.lora_request = LoRARequest(
                    lora_id=1,
                    lora_path=lora_path,
                    lora_int_id=1
                )
                
                self.current_lora_path = lora_path
                self.inference_stats['abliteration_active'] = True
                self.inference_stats['lora_switches'] += 1
                
                print("* Abliteration LoRA applied successfully to VLLM engine")
                return True
            
        except Exception as e:
            print(f"* [red]Failed to apply abliteration LoRA: {e}[/]")
        
        return False
    
    def _create_abliteration_lora(self, abliteration_params: Dict[str, Any]) -> Optional[str]:
        """
        Create a LoRA adapter from abliteration parameters.
        
        This is a simplified implementation - in practice, this would require
        converting the abliteration weights into LoRA format.
        
        Args:
            abliteration_params: Abliteration parameters
            
        Returns:
            Path to created LoRA adapter or None if failed
        """
        # This is a placeholder for the actual LoRA creation logic
        # In a full implementation, this would:
        # 1. Extract abliteration weights from the parameters
        # 2. Convert them to LoRA format (A and B matrices)
        # 3. Save as a proper LoRA adapter directory structure
        # 4. Return the path
        
        print("* [yellow]LoRA abliteration creation not fully implemented[/]")
        print("* This would require converting abliteration weights to LoRA format")
        
        # For now, return None to indicate not implemented
        return None
    
    def _remove_current_lora(self):
        """Remove current LoRA adapter."""
        if self.lora_request:
            self.lora_request = None
            self.current_lora_path = None
            self.inference_stats['abliteration_active'] = False
            print("* Previous abliteration LoRA removed")
    
    def disable_abliteration(self):
        """Disable abliteration (remove LoRA adapter)."""
        self._remove_current_lora()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self.inference_stats.copy()
        stats.update({
            'vllm_enabled': self.is_initialized,
            'tensor_parallel_size': self.vllm_tensor_parallel_size,
            'batch_size': self.vllm_batch_size,
            'gpu_memory_utilization': self.vllm_gpu_memory_utilization,
            'model_path': self.model_path,
        })
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_stats = {
            'total_requests': 0,
            'total_time': 0.0,
            'tokens_per_second': 0.0,
            'batches_processed': 0
        }
    
    def cleanup(self):
        """Clean up VLLM engine and free memory."""
        if self.vllm_engine is not None:
            del self.vllm_engine
            self.vllm_engine = None
        
        # Clean up LoRA cache
        import shutil
        if os.path.exists(self.abliteration_cache_dir):
            shutil.rmtree(self.abliteration_cache_dir, ignore_errors=True)
        
        self.is_initialized = False
        self.lora_request = None
        self.current_lora_path = None
        empty_cache()
        print("* VLLM inference engine cleaned up")


class VLLMEvaluator:
    """
    VLLM-enhanced evaluator for optimized refusal counting.
    
    Enhanced with on-the-fly abliteration support through LoRA adapters.
    """
    
    def __init__(self, settings: Settings, model_path: str = None):
        """
        Initialize VLLM evaluator.
        
        Args:
            settings: Heretic settings object
            model_path: Path to model (defaults to settings.model)
        """
        self.settings = settings
        self.vllm_wrapper = VLLMInferenceWrapper(settings, model_path)
        self.fallback_to_standard = False
        self.current_abliteration_params = None
        
    def initialize(self) -> bool:
        """
        Initialize VLLM evaluator.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        success = self.vllm_wrapper.initialize()
        if not success:
            self.fallback_to_standard = True
            print("* Will use standard inference for refusal counting")
        return success
    
    def count_refusals_vllm(self, bad_prompts: List[str], use_abliteration: bool = False) -> int:
        """
        Count refusals using VLLM for maximum performance.
        
        Args:
            bad_prompts: List of prompts that should trigger refusals
            use_abliteration: Whether to apply abliteration (if available)
            
        Returns:
            Number of refusals detected
        """
        if self.fallback_to_standard:
            raise RuntimeError("VLLM not available, use standard evaluator instead")
        
        print("* Using VLLM tensor parallel inference for refusal counting...")
        if use_abliteration and self.vllm_wrapper.inference_stats['abliteration_active']:
            print("* Using on-the-fly abliteration via LoRA")
        print(f"* Processing {len(bad_prompts)} prompts with batch size {self.vllm_wrapper.vllm_batch_size}")
        
        # Generate responses using VLLM
        responses = self.vllm_wrapper.get_responses_batched(bad_prompts, use_abliteration=use_abliteration)
        
        # Count refusals using the same logic as standard evaluator
        refusals = 0
        for response in responses:
            if self._is_refusal(response):
                refusals += 1
        
        # Print performance stats
        stats = self.vllm_wrapper.get_performance_stats()
        if stats['tokens_per_second'] > 0:
            print(f"* VLLM performance: {stats['tokens_per_second']:.0f} tokens/s")
        
        return refusals
    
    def apply_abliteration(self, abliteration_params: Dict[str, Any]) -> bool:
        """
        Apply abliteration parameters for on-the-fly modification.
        
        Args:
            abliteration_params: Abliteration parameters from heretic
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.current_abliteration_params = abliteration_params
        return self.vllm_wrapper.apply_abliteration_lora(abliteration_params)
    
    def disable_abliteration(self):
        """Disable abliteration."""
        self.vllm_wrapper.disable_abliteration()
        self.current_abliteration_params = None
    
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get VLLM performance statistics."""
        return self.vllm_wrapper.get_performance_stats()
    
    def cleanup(self):
        """Clean up VLLM evaluator."""
        self.vllm_wrapper.cleanup()


def create_vllm_evaluator(settings: Settings, model_path: str = None) -> VLLMEvaluator:
    """
    Factory function to create VLLM evaluator.
    
    Args:
        settings: Heretic settings object
        model_path: Path to model (optional)
        
    Returns:
        VLLMEvaluator instance
    """
    evaluator = VLLMEvaluator(settings, model_path)
    evaluator.initialize()
    return evaluator