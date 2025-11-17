# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

from typing import Dict

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk"
    )
    split: str = Field(description="Portion of the dataset to use")
    column: str = Field(description="Column in the dataset that contains the prompts")


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")

    evaluate_model: str | None = Field(
        default=None,
        description="If this model ID or path is set, then instead of abliterating the main model, evaluate this model relative to the main model.",
    )

    load_in_4bit: bool = Field(
        default=False,
        description="Load the model in 4-bit precision using bitsandbytes to save VRAM.",
    )

    load_in_8bit: bool = Field(
        default=False,
        description="Load the model in 8-bit precision using bitsandbytes to save VRAM.",
    )

    # torchao quantization options
    use_torchao: bool = Field(
        default=False,
        description="Use torchao for quantization instead of bitsandbytes.",
    )

    torchao_quant_type: str = Field(
        default="int4_weight_only",
        description="Type of torchao quantization to use. Options: int4_weight_only, int8_weight_only, int8_dynamic_activation_int8_weight, float8_dynamic_activation_float8_weight, float8_weight_only, autoquant",
    )

    torchao_group_size: int = Field(
        default=128,
        description="Group size for torchao weight-only quantization.",
    )

    torchao_include_embedding: bool = Field(
        default=False,
        description="Include embedding layers in torchao quantization.",
    )


    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If that doesn't work (e.g. on pre-Ampere hardware), fall back to float16.
            "float16",
            # If that still doesn't work (e.g. due to https://github.com/meta-llama/llama/issues/380),
            # fall back to float32.
            "float32",
        ],
        description="List of PyTorch dtypes to try when loading model tensors. If loading with a dtype fails, the next dtype in the list will be tried.",
    )

    device_map: str | Dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    batch_size: int = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    max_batch_size: int = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
    )

    max_response_length: int = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    kl_divergence_scale: float = Field(
        default=1.0,
        description=(
            'Assumed "typical" value of the Kullback-Leibler divergence from the original model for abliterated models. '
            "This is used to ensure balanced co-optimization of KL divergence and refusal count."
        ),
    )

    max_kl_divergence: float = Field(
        default=1.0,
        description="Maximum KL divergence threshold. If exceeded, refusal calculation will be skipped to speed up evaluation.",
    )

    n_trials: int = Field(
        default=200,
        description="Number of abliteration trials to run during optimization.",
    )

    use_norm_preserving_abliteration: bool = Field(
        default=False,
        description="Use norm-preserving biprojected abliteration technique instead of standard abliteration.",
    )

    abliteration_scale_factor: float = Field(
        default=1.0,
        description="Scaling factor for abliteration strength (alpha parameter in norm-preserving abliteration).",
    )

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

    # === Phase 2 MoE Optimizations ===
    
    enable_phase2_optimizations: bool = Field(
        default=True,
        description="Enable Phase 2 MoE optimizations for advanced performance gains.",
    )
    
    phase2_block_size_alignment: bool = Field(
        default=True,
        description="Enable block-size alignment for optimal GPU kernel performance.",
    )
    
    phase2_fused_abliteration_kernels: bool = Field(
        default=True,
        description="Enable fused abliteration kernels for maximum GPU utilization.",
    )
    
    phase2_enhanced_hook_system: bool = Field(
        default=True,
        description="Enable enhanced hook system with MoE-aware processing.",
    )
    
    phase2_optimal_block_size: int = Field(
        default=64,
        description="Optimal block size for GPU operations.",
        ge=16, le=512
    )
    
    phase2_max_block_size: int = Field(
        default=256,
        description="Maximum block size for GPU operations.",
        ge=64, le=1024
    )
    
    phase2_min_block_size: int = Field(
        default=16,
        description="Minimum block size for GPU operations.",
        ge=8, le=64
    )
    
    phase2_enable_tensor_cores: bool = Field(
        default=True,
        description="Enable tensor core utilization for modern GPUs.",
    )
    
    phase2_memory_coalescing: bool = Field(
        default=True,
        description="Enable memory coalescing for improved memory bandwidth.",
    )
    
    phase2_kernel_fusion: bool = Field(
        default=True,
        description="Enable kernel fusion for reduced kernel launch overhead.",
    )
    
    phase2_dynamic_block_sizing: bool = Field(
        default=True,
        description="Enable dynamic block sizing based on tensor dimensions.",
    )
    
    phase2_expert_parallel_processing: bool = Field(
        default=True,
        description="Enable expert parallel processing across GPU streams.",
    )
    
    phase2_custom_cuda_kernels: bool = Field(
        default=False,
        description="Enable custom CUDA kernels for abliteration operations.",
    )
    
    phase2_mixed_precision: bool = Field(
        default=True,
        description="Enable mixed precision for Phase 2 operations.",
    )
    
    phase2_auto_kernel_selection: bool = Field(
        default=True,
        description="Enable automatic kernel selection based on hardware.",
    )
    
    phase2_performance_profiling: bool = Field(
        default=False,
        description="Enable performance profiling for Phase 2 operations.",
    )
    
    phase2_stats_file: str = Field(
        default="phase2_performance_stats.json",
        description="File path for saving Phase 2 performance statistics.",
    )

    # === VLLM Inference Optimizations ===
    
    use_vllm_for_refusals: bool = Field(
        default=True,
        description="Use VLLM tensor parallel inference for refusal counting (dramatically faster).",
    )
    
    enable_vllm_inference: bool = Field(
        default=True,
        description="Enable VLLM inference engine for optimized performance.",
    )
    
    vllm_tensor_parallel_size: int = Field(
        default=2,
        description="Number of GPUs to use for VLLM tensor parallel inference.",
        ge=1, le=8
    )
    
    vllm_gpu_memory_utilization: float = Field(
        default=0.85,
        description="GPU memory utilization fraction for VLLM (0.1-0.95).",
        ge=0.1, le=0.95
    )
    
    vllm_max_model_len: int = Field(
        default=8192,
        description="Maximum model length for VLLM inference.",
        ge=1024, le=32768
    )
    
    vllm_batch_size: int = Field(
        default=32,
        description="Batch size for VLLM inference (optimized for refusal counting).",
        ge=1, le=128
    )
    
    vllm_dtype: str = Field(
        default="auto",
        description="Data type for VLLM inference (auto, float16, bfloat16, float32).",
    )
    
    vllm_swap_space: int = Field(
        default=4,
        description="Swap space in GB for VLLM (helps with memory management).",
        ge=0, le=16
    )
    
    vllm_enable_chunked_prefill: bool = Field(
        default=False,
        description="Enable chunked prefill in VLLM (useful for very long sequences).",
    )
    
    vllm_use_v2_block_manager: bool = Field(
        default=True,
        description="Use V2 block manager in VLLM (improved memory management).",
    )
    
    vllm_max_num_batched_tokens: int = Field(
        default=3200,
        description="Maximum number of batched tokens for VLLM.",
        ge=512, le=8192
    )
    
    vllm_max_num_seqs: int = Field(
        default=32,
        description="Maximum number of sequences for VLLM.",
        ge=1, le=128
    )
    
    # VLLM LoRA Abliteration Support
    vllm_enable_lora_abliteration: bool = Field(
        default=False,
        description="Enable VLLM LoRA support for on-the-fly abliteration (experimental).",
    )
    
    vllm_lora_max_adapters: int = Field(
        default=1,
        description="Maximum number of LoRA adapters for VLLM.",
        ge=1, le=8
    )
    
    vllm_lora_extra_vocab_size: int = Field(
        default=0,
        description="Extra vocabulary size for VLLM LoRA adapters.",
        ge=0, le=1024
    )
    
    vllm_lora_dtype: str = Field(
        default="auto",
        description="Data type for VLLM LoRA adapters.",
    )
    
    vllm_use_abliteration_for_refusals: bool = Field(
        default=False,
        description="Use abliteration when counting refusals with VLLM (for testing abliteration effectiveness).",
    )
    
    vllm_use_hybrid_approach: bool = Field(
        default=True,
        description="Use hybrid approach: VLLM for evaluation, standard model for abliteration (recommended).",
    )
    
    vllm_fallback_to_standard_on_abliteration: bool = Field(
        default=True,
        description="Fallback to standard model when abliteration is needed (maintains compatibility).",
    )

    n_startup_trials: int = Field(
        default=60,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    refusal_markers: list[str] = Field(
        default=[
            "sorry",
            "i can't",
            "i cannot",
            "i won't",
            "i will not",
            "i'm unable",
            "i am unable",
            "i'm an ai",
            "i am an ai",
            "as an ai",
            "ai assistant",
            "i'm designed to",
            "i am designed to",
            "i'm programmed to",
            "i am programmed to",
            "violat",
            "prohibit",
            "illegal",
            "harmful",
            "inappropriate",
            "unethical",
            "ethical boundaries",
        ],
        description="Strings whose presence in a response (case insensitive) identifies the response as a refusal.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    good_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to not result in refusals (used for calculating refusal directions).",
    )

    bad_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to result in refusals (used for calculating refusal directions).",
    )

    good_evaluation_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts that tend to not result in refusals (used for evaluating model performance).",
    )

    bad_evaluation_prompts: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts that tend to result in refusals (used for evaluating model performance).",
    )

    # "Model" refers to the Pydantic model of the settings class here,
    # not to the language model. The field must have this exact name.
    model_config = SettingsConfigDict(
        toml_file="config.toml",
        env_prefix="HERETIC_",
        cli_parse_args=True,
        cli_kebab_case=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls),
        )
