# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

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


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class AbliterationHook:
    """Hook to apply abliteration on-the-fly during forward pass"""
    def __init__(self, model, refusal_directions, direction_index, parameters, good_residuals=None, bad_residuals=None):
        self.model = model
        self.refusal_directions = refusal_directions
        self.direction_index = direction_index
        self.parameters = parameters
        self.hooks = []
        self.use_norm_preserving = model.settings.use_norm_preserving_abliteration
        self.scale_factor = model.settings.abliteration_scale_factor
        self.good_residuals = good_residuals
        self.bad_residuals = bad_residuals
        
        # Pre-compute mean directions for projection if needed
        if self.use_norm_preserving and good_residuals is not None and bad_residuals is not None:
            self.harmful_mean = bad_residuals.mean(dim=0)
            self.harmless_mean = good_residuals.mean(dim=0)
        else:
            self.harmful_mean = None
            self.harmless_mean = None
        
    def __enter__(self):
        # Register hooks for all layers
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
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def make_hook_fn(self, layer_index, component):
        def hook_fn(module, input, output):
            # Get parameters for this component
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
            if self.direction_index is None:
                layer_refusal_direction = self.refusal_directions[layer_index + 1]
            else:
                weight_idx, idx = math.modf(self.direction_index + 1)
                layer_refusal_direction = F.normalize(
                    self.refusal_directions[int(idx)].lerp(
                        self.refusal_directions[int(idx) + 1],
                        weight_idx,
                    ),
                    p=2,
                    dim=0,
                )
            
            # Apply abliteration transformation on-the-fly
            # output shape: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            
            if self.use_norm_preserving:
                # Norm-preserving abliteration approach with biprojection
                device = output.device
                dtype = output.dtype
                
                # Apply biprojection: project refusal direction onto harmless direction from target layer
                if self.harmless_mean is not None:
                    # Get harmless direction from target layer
                    harmless_dir = self.harmless_mean[layer_index]
                    
                    # Normalize harmless direction
                    harmless_normalized = F.normalize(harmless_dir.float(), dim=0)
                    
                    # Project and subtract to refine refusal direction (biprojection)
                    projection_scalar = torch.dot(layer_refusal_direction, harmless_normalized)
                    refined_refusal_dir = layer_refusal_direction - projection_scalar * harmless_normalized
                    layer_refusal_direction = refined_refusal_dir
                
                refusal_normalized = F.normalize(layer_refusal_direction.to(device).to(dtype), dim=0)
                
                # Apply norm-preserving transformation to outputs
                if len(output.shape) == 3:
                    # For attention output (batch, seq, hidden)
                    original_shape = output.shape
                    output_reshaped = output.view(-1, output.shape[-1])
                    
                    # Apply norm-preserving transformation
                    output_modified = self._apply_norm_preserving_to_output(
                        output_reshaped, refusal_normalized, weight * self.scale_factor
                    )
                    
                    output = output_modified.view(original_shape)
                else:
                    # For MLP output (batch, hidden) or other shapes
                    output = self._apply_norm_preserving_to_output(
                        output, refusal_normalized, weight * self.scale_factor
                    )
            else:
                # Standard abliteration approach
                # Ensure projector is on the same device as output
                device = output.device
                projector = torch.outer(
                    layer_refusal_direction.to(device),
                    layer_refusal_direction.to(device),
                ).to(output.dtype)
                
                # Apply transformation: output - weight * (projector @ output)
                # This is equivalent to modifying the weight matrix
                if len(output.shape) == 3:
                    # For attention output (batch, seq, hidden)
                    # Reshape for batch processing: (batch*seq, hidden)
                    original_shape = output.shape
                    output_reshaped = output.view(-1, output.shape[-1])
                    
                    # Apply vectorized transformation
                    projection = torch.matmul(projector, output_reshaped.T).T
                    output_reshaped = output_reshaped - weight * projection
                    
                    # Reshape back to original
                    output = output_reshaped.view(original_shape)
                else:
                    # For MLP output (batch, hidden) or other shapes
                    # Apply vectorized transformation
                    projection = torch.matmul(projector, output.T).T
                    output = output - weight * projection
            
            return output
        
        return hook_fn
    
    def _apply_norm_preserving_to_output(self, output: torch.Tensor, refusal_normalized: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Apply norm-preserving transformation to output activations."""
        # Compute projection onto refusal direction
        projection_coeffs = torch.matmul(output, refusal_normalized)
        projection = torch.outer(projection_coeffs, refusal_normalized)
        
        # Apply norm-preserving transformation
        output_directional = output - scale_factor * projection
        output_norms = torch.norm(output, dim=-1, keepdim=True)
        output_directional_norms = torch.norm(output_directional, dim=-1, keepdim=True)
        
        # Avoid division by zero
        output_directional_norms = torch.clamp(output_directional_norms, min=1e-8)
        
        # Preserve original norms
        output_preserved = output_directional * (output_norms / output_directional_norms)
        
        return output_preserved


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        
        # Ensure tokenizer has chat template set for proper chat functionality
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # Set a default chat template if none exists
            if hasattr(self.tokenizer, 'default_chat_template'):
                self.tokenizer.chat_template = self.tokenizer.default_chat_template
            else:
                # Use a basic chat template as fallback
                self.tokenizer.chat_template = "{% for message in messages %}{% for message in messages %}"

        self.model = None
        self.abliteration_hook = None  # For on-the-fly abliteration
        self.abliteration_params = None  # Store abliteration parameters

        # Check if torchao quantization is requested
        if settings.use_torchao:
            print(f"* Loading model with torchao quantization ({settings.torchao_quant_type})... ", end="")
            try:
                # Create torchao quantization config
                quantization_config = self._create_torchao_config()
                
                # Load model with torchao quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    quantization_config=quantization_config,
                    device_map=settings.device_map,
                    torch_dtype="auto",
                )

                # A test run can reveal dtype-related problems
                self.generate(["Test"], max_new_tokens=1)
                print("[green]Ok[/]")
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                raise Exception(f"Failed to load model with torchao quantization: {error}")
        # Check if bitsandbytes quantization is requested
        elif settings.load_in_4bit or settings.load_in_8bit:
            print(f"* Loading model in {'4-bit' if settings.load_in_4bit else '8-bit'} precision... ", end="")
            try:
                # Load quantized model for optimization
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    load_in_4bit=settings.load_in_4bit,
                    load_in_8bit=settings.load_in_8bit,
                    device_map=settings.device_map,
                    torch_dtype=torch.bfloat16,  # Ensure we're using bfloat16 for computation
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Ensure 4-bit computation uses bfloat16
                )

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(["Test"], max_new_tokens=1)
                print("[green]Ok[/]")
            except Exception as error:
                self.model = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                raise Exception(f"Failed to load model with quantization: {error}")
        else:
            # Try different dtypes if quantization is not used
            for dtype in settings.dtypes:
                print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.model,
                        dtype=dtype,
                        device_map=settings.device_map,
                    )

                    # A test run can reveal dtype-related problems such as the infamous
                    # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                    # (https://github.com/meta-llama/llama/issues/380).
                    self.generate(["Test"], max_new_tokens=1)
                except Exception as error:
                    self.model = None
                    empty_cache()
                    print(f"[red]Failed[/] ({error})")
                    continue

                print("[green]Ok[/]")
                break

            if self.model is None:
                raise Exception("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(matrices)}[/] matrices per layer"
            )

    def _create_torchao_config(self):
        """Create a torchao quantization configuration based on settings."""
        try:
            # Import torchao quantization configs
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
                Int8DynamicActivationInt8WeightConfig,
                Float8DynamicActivationFloat8WeightConfig,
                Float8WeightOnlyConfig,
                ModuleFqnToConfig,
            )
            from torchao.dtypes import (
                Int4CPULayout,
                Int4XPULayout,
                MarlinSparseLayout,
            )
            from torchao.quantization.quant_primitives import ZeroPointDomain
        except ImportError as e:
            raise ImportError(f"torchao is not installed. Please install it with: pip install torchao transformers. Error: {e}")

        quant_type = self.settings.torchao_quant_type
        group_size = self.settings.torchao_group_size
        include_embedding = self.settings.torchao_include_embedding
        
        # Determine device-specific layout
        device_map = self.settings.device_map
        if isinstance(device_map, str):
            device = device_map
        else:
            # If device_map is a dict, assume first device is the target
            device = "cuda" if any("cuda" in str(v) for v in device_map.values()) else "cpu"
        
        # Create the appropriate quantization config
        if quant_type == "int4_weight_only":
            if device == "cpu":
                layout = Int4CPULayout()
            elif device == "xpu":
                layout = Int4XPULayout()
            else:  # CUDA
                layout = None  # Use default for CUDA
            
            if layout:
                quant_config = Int4WeightOnlyConfig(group_size=group_size, layout=layout)
            else:
                quant_config = Int4WeightOnlyConfig(group_size=group_size)
                
        elif quant_type == "int8_weight_only":
            quant_config = Int8WeightOnlyConfig()
            
        elif quant_type == "int8_dynamic_activation_int8_weight":
            quant_config = Int8DynamicActivationInt8WeightConfig()
            
        elif quant_type == "float8_dynamic_activation_float8_weight":
            quant_config = Float8DynamicActivationFloat8WeightConfig()
            
        elif quant_type == "float8_weight_only":
            quant_config = Float8WeightOnlyConfig()
            
        elif quant_type == "int4_weight_only_sparse":
            layout = MarlinSparseLayout()
            quant_config = Int4WeightOnlyConfig(group_size=group_size, layout=layout)
            
        elif quant_type == "autoquant":
            # For autoquant, we pass the string directly
            return TorchAoConfig("autoquant", min_sqnr=None)
            
        else:
            raise ValueError(f"Unsupported torchao quantization type: {quant_type}")
        
        return TorchAoConfig(
            quant_type=quant_config,
            include_embedding=include_embedding
        )

    def reload_model(self):
        # Store dtype before clearing model
        dtype = self.model.dtype if self.model else None
        
        # Purge existing model object from memory to make space.
        self.model = None
        
        # Force garbage collection to ensure memory is freed
        import gc
        gc.collect()
        
        # Clear ALL CUDA caches and reset memory stats
        if torch.cuda.is_available():
            # Clear cache for ALL GPUs
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    # Force additional cleanup
                    torch.cuda.ipc_collect()
        
        # Call empty_cache for comprehensive cleanup
        empty_cache()

        # For all models, we use on-the-fly abliteration which doesn't require model reloading
        if self.model is not None:
            print("* Model already loaded - using on-the-fly abliteration, no reload needed")
            return  # Skip reloading entirely
        else:
            # Only reload if model is None (initial load or after explicit clearing)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model,
                dtype=dtype,
                device_map=self.settings.device_map,
                low_cpu_mem_usage=True,  # Enable to reduce memory usage during loading
            )

    def get_layers(self) -> ModuleList:
        # Most multimodal models.
        with suppress(Exception):
            return self.model.model.language_model.layers

        # Text-only models.
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            assert torch.is_tensor(matrix)

            if component not in matrices:
                matrices[component] = []

            matrices[component].append(matrix)

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        # gpt-oss MoE.
        with suppress(Exception):
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # GLM MoE models.
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)
        # GLM MoE shared experts.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_experts.down_proj.weight)

        # We need at least one MLP down-projection.
        assert matrices["mlp.down_proj"]

        return matrices

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_matrices(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ):
        # Apply abliteration using on-the-fly approach for all models
        if self.settings.use_norm_preserving_abliteration:
            print(f"* Using on-the-fly norm-preserving biprojected abliteration (scale factor: {self.settings.abliteration_scale_factor})... ", end="")
        else:
            print("* Using on-the-fly standard abliteration (fast and compatible)... ", end="")
        try:
            # Store abliteration parameters for on-the-fly application
            self.abliteration_params = {
                'refusal_directions': refusal_directions,
                'direction_index': direction_index,
                'parameters': parameters,
                'good_residuals': good_residuals,
                'bad_residuals': bad_residuals
            }
            print("[green]Ok[/]")
            print("* No model reloading needed for subsequent trials - using on-the-fly transformation")
            # Store parameters for on-the-fly application only
        except Exception as error:
            print(f"[red]Failed[/] ({error})")
            print("* Falling back to CPU-based abliteration...")
            self._abliterate_via_cpu(refusal_directions, direction_index, parameters)

    def apply_final_abliteration(self, refusal_directions: Tensor, direction_index: float | None, parameters: dict[str, AbliterationParameters]):
        """Apply abliteration to model weights for final use (saving only)."""
        if self.settings.use_norm_preserving_abliteration:
            print(f"* Applying final norm-preserving projected abliteration (scale factor: {self.settings.abliteration_scale_factor})")
        else:
            print("* Applying final standard abliteration to model weights for saving...")
            
        if self.settings.use_torchao or self.settings.load_in_4bit or self.settings.load_in_8bit:
            # For quantized models, use CPU-based processing to save unquantized model
            print("* Quantized model detected - loading full precision model to CPU for final abliteration...")
            self._abliterate_via_cpu(refusal_directions, direction_index, parameters)
        else:
            # For unquantized models, apply directly to weights in VRAM
            print("* Unquantized model detected - applying abliteration directly to weights in VRAM...")
            # Note: We don't have residuals here either - this is a design limitation
            # The original implementation stores measurements separately and loads them during abliteration
            self._apply_abliteration_to_model(self.model, refusal_directions, direction_index, parameters)

    def _abliterate_via_cpu(self, refusal_directions: Tensor, direction_index: float | None, parameters: dict[str, AbliterationParameters]):
        """Abliterate by loading full precision model to CPU, applying changes, then loading back to device."""
        import tempfile
        import os
        
        # Clear current model from VRAM before loading full precision model
        self.model = None
        
        # Aggressive memory cleanup
        empty_cache()
        
        # Load full precision model to CPU RAM with memory optimizations
        print("* Loading full precision model to CPU RAM for abliteration...")
        full_model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,  # Enable to reduce memory usage
        )
        
        # Apply abliteration to full precision model
        print("* Applying abliteration to full precision model in CPU...")
        # Note: We don't have residuals here, so projection won't be applied for CPU path
        # This is a limitation - ideally we'd store residuals for later use
        self._apply_abliteration_to_model(full_model, refusal_directions, direction_index, parameters)
        
        # Save the abliterated full precision model to a temporary location
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, "temp_model")
        
        try:
            # Save with memory-efficient options
            print("* Saving abliterated full precision model...")
            full_model.save_pretrained(
                temp_model_path,
                safe_serialization=False,  # torchao requires safe_serialization=False
                max_shard_size="2GB",  # Split into smaller shards to reduce memory pressure
            )
           
            # Clean up the full model BEFORE loading the final model
            del full_model
            full_model = None
            empty_cache()
           
            # Load the abliterated model WITHOUT quantization (save as full precision)
            print("* Loading final abliterated model (full precision) to target device...")
            self.model = AutoModelForCausalLM.from_pretrained(
                temp_model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.settings.device_map,
                low_cpu_mem_usage=True,  # Enable to reduce memory usage
            )
            print("* Full precision abliterated model loaded successfully")
        finally:
            # Ensure temporary directory is always cleaned up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Final cleanup
        empty_cache()

    def load_quantized_model(self):
        """Load the quantized model from the original model (for original model selection)"""
        if self.settings.use_torchao or self.settings.load_in_4bit or self.settings.load_in_8bit:
            # Save the current model to a temporary location
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            temp_model_path = os.path.join(temp_dir, "temp_model")
            self.model.save_pretrained(
                temp_model_path,
                safe_serialization=False,  # torchao requires safe_serialization=False
            )
            
            # Clear current model from memory
            self.model = None
            empty_cache()
            
            # Load the model with quantization
            if self.settings.use_torchao:
                # Load with torchao quantization
                quantization_config = self._create_torchao_config()
                self.model = AutoModelForCausalLM.from_pretrained(
                    temp_model_path,
                    quantization_config=quantization_config,
                    device_map=self.settings.device_map,
                    torch_dtype="auto",
                )
            else:
                # Load with bitsandbytes quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    temp_model_path,
                    load_in_4bit=self.settings.load_in_4bit,
                    load_in_8bit=self.settings.load_in_8bit,
                    device_map=self.settings.device_map,
                    torch_dtype=torch.bfloat16,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)

    def compute_projected_directions(self, refusal_directions: Tensor, good_residuals: Tensor, bad_residuals: Tensor) -> Tensor:
        """Compute projected refusal directions using Gram-Schmidt orthogonalization as in the original implementation."""
        # Compute mean directions for harmful and harmless
        harmful_mean = bad_residuals.mean(dim=0)
        harmless_mean = good_residuals.mean(dim=0)
        
        projected_directions = []
        
        for layer_idx in range(refusal_directions.shape[0]):
            harmful_layer = harmful_mean[layer_idx]
            harmless_layer = harmless_mean[layer_idx]
            
            # Initial refusal direction
            refusal_dir = harmful_layer - harmless_layer
            
            # Apply Gram-Schmidt orthogonalization (project out harmless direction)
            # Normalize harmless direction to avoid numerical issues
            harmless_normalized = F.normalize(harmless_layer.float(), dim=0)
            
            # Project and subtract contribution along harmless direction
            projection_scalar = torch.dot(refusal_dir, harmless_normalized)
            
            # Resulting refusal direction should minimize impact along harmless direction
            refusal_dir = refusal_dir - projection_scalar * harmless_normalized
            
            # Normalize the final direction
            refusal_dir = F.normalize(refusal_dir, dim=0)
            
            projected_directions.append(refusal_dir)
        
        return torch.stack(projected_directions)

    def _norm_preserving_abliterate_impl(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ):
        """Implement norm-preserving abliteration algorithm matching the original implementation."""
        scale_factor = self.settings.abliteration_scale_factor
        
        # Compute mean directions for projection (needed for on-the-fly projection)
        if good_residuals is not None and bad_residuals is not None:
            harmful_mean = bad_residuals.mean(dim=0)
            harmless_mean = good_residuals.mean(dim=0)
        else:
            # Fallback if residuals not provided
            harmful_mean = None
            harmless_mean = None

        for layer_index in range(len(self.get_layers())):
            for component, matrices in self.get_layer_matrices(layer_index).items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                # Get measurement layer index based on direction_index
                if direction_index is None:
                    measurement_layer = layer_index
                else:
                    weight_idx, idx = math.modf(direction_index)
                    measurement_layer = int(idx)

                # Get refusal direction from measurement layer
                refusal_dir = refusal_directions[measurement_layer + 1]  # +1 for embeddings
                
                # Apply biprojection: project refusal direction onto harmless direction from target layer
                if harmless_mean is not None:
                    # Get harmless direction from target layer
                    harmless_dir = harmless_mean[layer_index]
                    
                    # Normalize harmless direction
                    harmless_normalized = F.normalize(harmless_dir.float(), dim=0)
                    
                    # Project and subtract to refine refusal direction (biprojection)
                    projection_scalar = torch.dot(refusal_dir, harmless_normalized)
                    refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
                    refusal_dir = refined_refusal_dir

                for matrix in matrices:
                    # Apply the norm-preserving modification matching the original implementation
                    modified_matrix = self.modify_tensor_norm_preserved(
                        matrix, refusal_dir, scale_factor * weight
                    )
                    
                    # Update the matrix
                    if hasattr(matrix, 'bnb_quantized') and matrix.bnb_quantized:
                        # For quantized matrices, try to update if possible
                        try:
                            if hasattr(matrix, 'weight') and matrix.weight is not None:
                                with torch.no_grad():
                                    matrix.weight.data = modified_matrix
                            else:
                                print(f"* Warning: Cannot update quantized matrix of type {type(matrix)}")
                        except Exception as e:
                            print(f"* Warning: Failed to update quantized matrix: {e}")
                    elif hasattr(matrix, '_data_impl') and hasattr(matrix._data_impl, '_value'):
                        # For torchao quantized weights
                        with torch.no_grad():
                            matrix.copy_(modified_matrix)
                    else:
                        # For regular weights
                        with torch.no_grad():
                            matrix.copy_(modified_matrix)

    def modify_tensor_norm_preserved(
        self, W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        Modify weight tensor by ablating refusal direction while preserving row norms.
        Adapted for PyTorch models (weights already in [out_features, in_features] format).
        """
        original_dtype = W.dtype
        device = W.device

        with torch.no_grad():
            # PyTorch models already have weights in [out_features, in_features] format
            # No transpose needed - this is different from the original that works with safetensors
            W_gpu = W.to(device, dtype=torch.float32)
            refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32)

            # Ensure refusal_dir is a 1-dimensional tensor
            if refusal_dir_gpu.dim() > 1:
                refusal_dir_gpu = refusal_dir_gpu.view(-1)
            
            # Normalize refusal direction
            refusal_normalized = F.normalize(refusal_dir_gpu, dim=0)

            # Decompose weight matrix
            # W_gpu is [out_features, in_features] in PyTorch format
            W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
            W_direction = F.normalize(W_gpu, dim=1)  # normalized per output neuron
        
            # Apply abliteration to the DIRECTIONAL component
            # Compute dot product of each row with refusal direction
            projection = torch.matmul(W_direction, refusal_normalized)  # [out_features]
            
            # Subtract the projection using outer product
            W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
        
            # Re-normalize the adjusted direction
            W_direction_new = F.normalize(W_direction_new, dim=1)
        
            # Recombine: keep original magnitude, use new direction
            W_modified = W_norm * W_direction_new
            
            # Convert back to original dtype (no transpose needed for PyTorch)
            result = W_modified.to(original_dtype)

        return result.detach().clone()

    def _abliterate_impl(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, matrices in self.get_layer_matrices(layer_index).items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                # Projects any right-multiplied vector(s) onto the subspace
                # spanned by the refusal direction.
                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)

                for matrix in matrices:
                    # Ensure projector is on the same device as the matrix
                    projector_device = projector.to(matrix.device)
                    
                    # Apply abliteration to both quantized and non-quantized matrices
                    if hasattr(matrix, 'bnb_quantized') and matrix.bnb_quantized:
                        # For bitsandbytes 4-bit quantized weights
                        # Try different approaches to access quantized data
                        
                        # Method 1: Try to access quant_state if available
                        if hasattr(matrix, 'quant_state'):
                            quant_state = matrix.quant_state()
                            weight_data = quant_state['weight'].dequantize()
                        # Method 2: Try to dequantize directly
                        elif hasattr(matrix, 'dequantize'):
                            weight_data = matrix.dequantize()
                        # Method 3: Try to access weight parameter and dequantize
                        elif hasattr(matrix, 'weight') and hasattr(matrix.weight, 'dequantize'):
                            weight_data = matrix.weight.dequantize()
                        else:
                            # If we can't dequantize, skip this matrix
                            print(f"* Warning: Cannot dequantize matrix of type {type(matrix)}, skipping...")
                            continue
                        
                        # Calculate the projection in the dequantized space
                        projection = weight * (projector_device @ weight_data)
                        
                        # Apply the abliteration by subtracting the projection
                        modified_weight = weight_data - projection
                        
                        # For in-place abliteration, we need to update the quantized weights
                        # This is complex and may not work reliably with all bitsandbytes versions
                        try:
                            # Try to update the weight directly if possible
                            if hasattr(matrix, 'weight') and matrix.weight is not None:
                                with torch.no_grad():
                                    matrix.weight.data = modified_weight
                            else:
                                print(f"* Warning: Cannot update quantized matrix of type {type(matrix)}, in-place update may not work")
                        except Exception as e:
                            print(f"* Warning: Failed to update quantized matrix: {e}")
                    elif hasattr(matrix, '_data_impl') and hasattr(matrix._data_impl, '_value'):
                        # For torchao quantized weights
                        # torchao uses a different quantization scheme with tensor subclasses
                        # We need to work with the underlying data
                        
                        # Get the dequantized weight data
                        weight_data = matrix.dequantize()
                        
                        # Calculate the projection in the dequantized space
                        projection = weight * (projector_device @ weight_data)
                        
                        # Apply the abliteration by subtracting the projection
                        modified_weight = weight_data - projection
                        
                        # For torchao, we need to update the weight data directly
                        # This is a simplified approach - in practice, torchao quantization
                        # might require more complex handling
                        with torch.no_grad():
                            # Update the tensor data in place
                            matrix.copy_(modified_weight)
                    else:
                        # For regular (non-quantized) weights, use the original approach
                        # In-place subtraction is safe as we're not using Autograd.
                        matrix.sub_(weight * (projector_device @ matrix))

    def _apply_abliteration_to_model(
        self,
        target_model,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        good_residuals: Tensor = None,
        bad_residuals: Tensor = None,
    ):
        """Apply abliteration to a different model instance (e.g., a full precision model)"""
        # Temporarily replace self.model with target_model for the duration of this operation
        original_model = self.model
        self.model = target_model
        
        try:
            # Apply the abliteration using the appropriate method
            if self.settings.use_norm_preserving_abliteration:
                self._norm_preserving_abliterate_impl(refusal_directions, direction_index, parameters, good_residuals, bad_residuals)
            else:
                self._abliterate_impl(refusal_directions, direction_index, parameters)
        finally:
            # Restore the original model
            self.model = original_model

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        # Check if the tokenizer supports system role
        try:
            # Try to apply chat template with system role to check if it's supported
            test_chat = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
            ]
            self.tokenizer.apply_chat_template(test_chat, tokenize=False)
            # If no exception, system role is supported
            return [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": prompt},
            ]
        except Exception:
            # If system role is not supported, prepend it to the user message
            modified_prompt = f"{self.settings.system_prompt}\n\n{prompt}"
            return [
                {"role": "user", "content": modified_prompt},
            ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # Apply on-the-fly abliteration if parameters are available
        if self.abliteration_params is not None:
            with AbliterationHook(
                self,
                self.abliteration_params['refusal_directions'],
                self.abliteration_params['direction_index'],
                self.abliteration_params['parameters'],
                self.abliteration_params.get('good_residuals'),
                self.abliteration_params.get('bad_residuals')
            ):
                return inputs, self.model.generate(
                    **inputs,
                    **kwargs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
                )
        else:
            return inputs, self.model.generate(
                **inputs,
                **kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
            )

    def get_responses(self, prompts: list[str]) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        # Return only the newly generated part.
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def get_responses_batched(self, prompts: list[str]) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(batch):
                responses.append(response)
            # Clear memory after each batch to prevent accumulation
            empty_cache()

        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Hidden states for the first (only) generated token.
        hidden_states = outputs.hidden_states[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))
            # Clear memory after each batch to prevent accumulation
            empty_cache()

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Logits for the first (only) generated token.
        logits = outputs.scores[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))
            # Clear memory after each batch to prevent accumulation
            empty_cache()

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # Check if the tokenizer supports system role
        try:
            # Try to apply chat template with system role to check if it's supported
            test_chat = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
            ]
            self.tokenizer.apply_chat_template(test_chat, tokenize=False)
            # If no exception, use the chat as is
            formatted_chat = chat
        except Exception:
            # If system role is not supported, append system content to first user message
            formatted_chat = []
            system_content = None
            
            # First, extract system content if present
            for message in chat:
                if message["role"] == "system":
                    system_content = message["content"]
                    break
            
            # Then process the rest of the messages
            for message in chat:
                if message["role"] == "system":
                    # Skip system messages as we'll handle them separately
                    continue
                elif message["role"] == "user":
                    if system_content is not None and not formatted_chat:
                        # This is the first user message and we have system content
                        # Append system content to the first user message
                        modified_content = f"{system_content}\n\n{message['content']}"
                        formatted_chat.append({"role": "user", "content": modified_content})
                        system_content = None  # Mark as used
                    else:
                        formatted_chat.append(message)
                else:
                    formatted_chat.append(message)
            
            # If we ended up with an empty chat (unlikely), add a default user message
            if not formatted_chat:
                formatted_chat = [{"role": "user", "content": "Hello"}]

        chat_prompt: str = self.tokenizer.apply_chat_template(
            formatted_chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Apply on-the-fly abliteration if parameters are available
        if self.abliteration_params is not None:
            with AbliterationHook(
                self,
                self.abliteration_params['refusal_directions'],
                self.abliteration_params['direction_index'],
                self.abliteration_params['parameters']
            ):
                outputs = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=4096,
                )
        else:
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=4096,
            )

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
