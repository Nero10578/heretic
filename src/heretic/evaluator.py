# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import torch.nn.functional as F

from .config import Settings
from .model import Model
from .utils import load_prompts, print
from .vllm_inference import create_vllm_evaluator


class Evaluator:
    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model
        
        # Initialize VLLM abliteration model for optimized evaluation
        self.vllm_abliteration_model = None
        self.use_vllm_for_refusals = getattr(settings, 'use_vllm_for_refusals', True)
        
        if self.use_vllm_for_refusals:
            print("* Initializing VLLM with on-the-fly abliteration support...")
            try:
                from .vllm_abliteration import create_vllm_abliteration_model
                self.vllm_abliteration_model = create_vllm_abliteration_model(settings)
                if self.vllm_abliteration_model:
                    print("* VLLM initialized successfully with LoRA abliteration support")
                    print("* Single model instance - no VRAM duplication")
                else:
                    print("* VLLM initialization failed, using standard model")
                    self.vllm_abliteration_model = None
            except Exception as e:
                print(f"* [yellow]Failed to initialize VLLM abliteration: {e}[/]")
                print("* Using standard model for all operations")
                self.vllm_abliteration_model = None

        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset}[/]..."
        )
        self.good_prompts = load_prompts(settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded")

        print("* Obtaining first-token probability distributions...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset}[/]..."
        )
        self.bad_prompts = load_prompts(settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded")

        print("* Counting model refusals...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)}"
        )

    def is_refusal(self, response: str) -> bool:
        # Remove emphasis (e.g. "I *will not*...").
        response = response.lower().replace("*", "")

        # Normalize typographic apostrophes ("won’t" -> "won't").
        response = response.replace("’", "'")

        for marker in self.settings.refusal_markers:
            if marker.lower() in response:
                return True

        return False

    def count_refusals(self, use_abliteration: bool = False) -> int:
        """
        Count refusals using VLLM with on-the-fly abliteration if available.
        
        Args:
            use_abliteration: Whether to use abliteration (if available)
            
        Returns:
            Number of refusals detected
        """
        if self.vllm_abliteration_model is not None:
            # Use VLLM with on-the-fly abliteration
            try:
                if use_abliteration and hasattr(self.model, 'abliteration_params') and self.model.abliteration_params:
                    # Apply abliteration to VLLM via LoRA
                    abliteration_params = self.model.abliteration_params
                    success = self.vllm_abliteration_model.apply_abliteration(
                        abliteration_params['refusal_directions'],
                        abliteration_params['direction_index'],
                        abliteration_params['parameters'],
                        abliteration_params.get('good_residuals'),
                        abliteration_params.get('bad_residuals')
                    )
                    
                    if success:
                        print("* Using VLLM with on-the-fly abliteration (LoRA)")
                    else:
                        print("* VLLM abliteration failed, using VLLM without abliteration")
                        use_abliteration = False
                else:
                    print("* Using VLLM without abliteration")
                
                # Generate responses with VLLM
                responses = self.vllm_abliteration_model.get_responses_batched(self.bad_prompts)
                
                # Count refusals
                refusals = len([r for r in responses if self.is_refusal(r)])
                
                # Print performance stats
                stats = self.vllm_abliteration_model.get_performance_stats()
                if stats.get('tokens_per_second', 0) > 0:
                    print(f"* VLLM performance: {stats['tokens_per_second']:.0f} tokens/s")
                
                return refusals
                
            except Exception as e:
                print(f"* [yellow]VLLM inference failed: {e}[/]")
                print("* Falling back to standard inference")
                self.vllm_abliteration_model = None
        
        # Fallback to standard inference
        if use_abliteration and hasattr(self.model, 'abliteration_params') and self.model.abliteration_params:
            print("* Using standard model with on-the-fly abliteration")
        else:
            print("* Using standard model without abliteration")
            
        responses = self.model.get_responses_batched(self.bad_prompts)
        refusals = len([r for r in responses if self.is_refusal(response)])
        return refusals

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        # Clear memory before evaluation to prevent OOM
        from .utils import empty_cache
        empty_cache()
        
        print("  * Obtaining first-token probability distributions...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        
        # Clear memory after getting logprobs
        empty_cache()
        
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.2f}[/]")

        # Check if KL divergence exceeds the maximum threshold
        if kl_divergence > self.settings.max_kl_divergence:
            print(f"  * [yellow]KL divergence exceeds max threshold ({self.settings.max_kl_divergence:.2f}), skipping refusal calculation[/]")
            refusals = self.base_refusals  # Use base refusals as fallback
            print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)} [grey50](skipped)[/]")
        else:
            # Clear memory before counting refusals
            empty_cache()
            
            print("  * Counting model refusals...")
            if self.vllm_evaluator is not None:
                print("  * Using VLLM tensor parallel inference for maximum speed...")
                if self.use_hybrid_approach:
                    print("  * Hybrid approach: VLLM for evaluation, standard model for abliteration")
            
            # Check if abliteration is available
            has_abliteration = (
                hasattr(self.model, 'abliteration_params') and
                self.model.abliteration_params is not None
            )
            
            if has_abliteration:
                print("  * Abliteration parameters available - will test effectiveness")
                refusals = self.count_refusals(use_abliteration=True)
            else:
                refusals = self.count_refusals(use_abliteration=False)
            
            print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")
            
            # Print VLLM performance stats if available
            if self.vllm_evaluator is not None:
                try:
                    stats = self.vllm_evaluator.get_performance_stats()
                    if stats.get('tokens_per_second', 0) > 0:
                        print(f"  * VLLM performance: {stats['tokens_per_second']:.0f} tokens/s")
                        if stats.get('abliteration_active', False):
                            print("  * VLLM abliteration: [green]Active[/]")
                        else:
                            print("  * VLLM abliteration: [grey50]Inactive[/]")
                except Exception:
                    pass

        score = (
            (kl_divergence / self.settings.kl_divergence_scale),
            (refusals / self.base_refusals),
        )

        return score, kl_divergence, refusals
