# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Arbitrary-Rank Ablation (ARA) (Weidmann 2026)
# See https://github.com/p-e-w/heretic/pull/211 for more information.

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import bitsandbytes.functional as BNB_F
import torch
import torch.linalg as LA
import torch.nn.functional as F
from optuna import Trial
from peft.tuners.lora.layer import Linear
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)
from torch import Tensor
from torch.optim import LBFGS

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.modifier import Context, Modifier, Serializable
from heretic.utils import REFUSAL_MARKERS, format_dataset_specification, print


@dataclass
class Parameters(Serializable):
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(**data)


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce desirable responses.",
    )

    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce undesirable responses.",
    )

    preserve_row_magnitudes: bool = Field(
        default=True,
        description=(
            "Whether to renormalize the rows of the modified matrices to preserve "
            "the original matrices' row magnitudes. This is believed to improve "
            'intelligence retention (see Lai 2025, "Magnitude-Preserving Orthogonal Ablation").'
        ),
    )

    lora_rank: PositiveInt = Field(
        default=50,
        description=(
            "The rank of the LoRA adapter to use. "
            'While mathematically, ARA is of "arbitrary" rank, experiments have shown that '
            "singular values tend to drop rapidly after a few dozen dimensions, and approximating "
            "the full transformation with a LoRA has many practical advantages."
        ),
    )

    n_optimization_steps: PositiveInt = Field(
        default=5,
        description="Number of (outer) L-BFGS optimization steps to perform.",
    )

    learning_rate: float = Field(
        default=1.0,
        description="Learning rate to use in the L-BFGS optimizer.",
    )

    max_iter: PositiveInt = Field(
        default=20,
        description="Maximum number of (inner) iterations to perform per (outer) L-BFGS optimization step.",
    )

    history_size: PositiveInt = Field(
        default=10,
        description="Number of past updates to store for approximating the Hessian matrix in the L-BFGS optimizer.",
    )

    print_loss: bool = Field(
        default=False,
        description="Whether to print the loss value for each L-BFGS optimization step.",
    )

    use_tf32: bool = Field(
        default=True,
        description=(
            "Run the (float32) inner optimization with TF32 tensor cores. The matmuls "
            "involve the full weight matrices and the prompt I/O, so without this a "
            "single kernel can run long enough to trip the CUDA launch timeout. TF32 "
            "only affects this optimization, not evaluation, and is several times faster."
        ),
    )

    filter_prompts: bool = Field(
        default=False,
        description=(
            "Whether to filter the prompt sets based on the model's refusal behavior "
            "before obtaining module I/O. When enabled, only harmful prompts that the "
            "model actually refuses are kept, and harmless prompts that are refused "
            "are discarded."
        ),
    )

    refusal_markers: list[str] = Field(
        default=REFUSAL_MARKERS,
        description=(
            "Keyword markers used to detect refusals when filtering prompts. "
            "A response is considered a refusal if it is empty or contains any of these "
            "strings (case insensitive). Only used when filter_prompts is enabled."
        ),
    )


# For each vector in the 2D-tensor `a`, computes the mean Euclidean distance
# to the `k` nearest neighbors of the vector among the vectors in the 2D-tensor `b`.
def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    distances = torch.cdist(a, b)
    nearest_distances, _ = distances.topk(k, dim=1, largest=False)
    return nearest_distances.mean(1)


# The objective function at the heart of ARA.
def ara_loss(
    good_output: Tensor,
    bad_output: Tensor,
    new_good_output: Tensor,
    new_bad_output: Tensor,
    parameters: Parameters,
) -> Tensor:
    # The outputs for "good" prompts should change as little as possible.
    preserve_good_behavior = ((new_good_output - good_output) ** 2).mean()

    steer_bad_behavior = (
        # Pull the outputs for "bad" prompts towards
        # the original outputs for "good" prompts.
        mean_distances_to_knn(
            new_bad_output,
            good_output,
            parameters.neighbor_count,
        ).mean()
        # Push the outputs for "bad" prompts away from
        # the original outputs for "bad" prompts.
        # In combination with the above, this overcorrects
        # away from the original residuals, which results
        # in stronger steering that can overcome more complex
        # refusal mechanisms.
        + parameters.overcorrect_relative_weight
        * -mean_distances_to_knn(
            new_bad_output,
            bad_output,
            parameters.neighbor_count,
        ).mean()
    )

    return (
        parameters.preserve_good_behavior_weight * preserve_good_behavior
        + parameters.steer_bad_behavior_weight * steer_bad_behavior
    )


class ARA(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        return "Arbitrary-Rank Ablation (ARA)"

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()

        cache = self.init_cache()
        if cache is not None:
            cached = cache.load_with_fallback(self.modifier_name)
            if cached is not None:
                self.good_prompt_count = cached["good_prompt_count"]
                self.bad_prompt_count = cached["bad_prompt_count"]
                self.good_module_io = cached["good_module_io"]
                self.bad_module_io = cached["bad_module_io"]
                print()
                print(
                    f"Reusing cached initialization for [bold]{self.modifier_name}[/]: "
                    f"[bold]{self.good_prompt_count}[/] good and "
                    f"[bold]{self.bad_prompt_count}[/] bad prompts."
                )
                model.apply_lora(self.settings.lora_rank)
                return

        # Capture module I/O while the reasoning is being generated (when
        # capture_during_generation is enabled) instead of re-forwarding the prompt plus
        # reasoning afterwards. The filtering below is the generation this reuses. ARA
        # only needs module I/O, so skip the (unused, large) logits capture.
        ctx.capture_module_io_default = True
        ctx.capture_logits_default = False

        print()
        print(
            f"Loading good prompts from [bold]{format_dataset_specification(self.settings.good_prompts)}[/]..."
        )
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

        print()
        print(
            f"Loading bad prompts from [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        if self.settings.filter_prompts:
            good_prompts, bad_prompts = self.filter_prompts_by_refusal(
                ctx, good_prompts, bad_prompts, self.settings.refusal_markers
            )

        self.good_prompt_count = len(good_prompts)
        self.bad_prompt_count = len(bad_prompts)

        print()
        print("Obtaining module I/O for good prompts...")
        self.good_module_io = ctx.get_module_io(good_prompts)

        print()
        print("Obtaining module I/O for bad prompts...")
        self.bad_module_io = ctx.get_module_io(bad_prompts)

        if cache is not None:
            cache.save(
                {
                    "good_prompt_count": self.good_prompt_count,
                    "bad_prompt_count": self.bad_prompt_count,
                    "good_module_io": self.good_module_io,
                    "bad_module_io": self.bad_module_io,
                }
            )

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.settings.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        layer_count = len(ctx.get_model().get_layers())

        start_layer_index = trial.suggest_int(
            "start_layer_index",
            0,
            layer_count // 2,
        )
        # The end index is exclusive, so starting its range one above the start range's
        # upper bound ensures that the layer range is never empty.
        end_layer_index = trial.suggest_int(
            "end_layer_index",
            layer_count // 2 + 1,
            layer_count,
        )
        preserve_good_behavior_weight = trial.suggest_float(
            "preserve_good_behavior_weight",
            0.0,
            1.0,
        )
        steer_bad_behavior_weight = trial.suggest_float(
            "steer_bad_behavior_weight",
            0.0001,
            1.0,
            log=True,
        )
        overcorrect_relative_weight = trial.suggest_float(
            "overcorrect_relative_weight",
            0.0,
            1.3,
        )
        # The nearest neighbors are selected from the outputs for the good and bad prompts,
        # which contain one vector per prompt, so there can't be more neighbors than prompts.
        neighbor_count = trial.suggest_int(
            "neighbor_count",
            1,
            min(15, self.good_prompt_count, self.bad_prompt_count),
        )

        return Parameters(
            start_layer_index=start_layer_index,
            end_layer_index=end_layer_index,
            preserve_good_behavior_weight=preserve_good_behavior_weight,
            steer_bad_behavior_weight=steer_bad_behavior_weight,
            overcorrect_relative_weight=overcorrect_relative_weight,
            neighbor_count=neighbor_count,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        # The optimization below works in float32 and multiplies the full weight matrices
        # by the prompt I/O. A single such matmul can run long enough to trip the CUDA
        # launch timeout, so use TF32 tensor cores for its duration. The synchronize at
        # the end makes such an error surface here instead of in a later, unrelated call.
        previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.settings.use_tf32
        try:
            self._modify_model(ctx, parameters)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32
            torch.cuda.synchronize()

    def _modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        # The model's reset zeroes both A and B, where all gradients vanish,
        # so A must be initialized below for the optimization to make progress.
        # Generating on the CPU keeps A independent of the device and the global RNG state.
        generator = torch.Generator().manual_seed(cast(int, self.heretic_settings.seed))

        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            for component, modules in model.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    # Cast to Linear to access weights and LoRA adapters.
                    module = cast(Linear, module)

                    # We optimize the LoRA weights A and B.
                    lora_A = cast(Tensor, module.lora_A["default"].weight)
                    lora_B = cast(Tensor, module.lora_B["default"].weight)

                    # The model can be parked in host memory (see Model.park) to leave the
                    # GPU to another engine, so run the optimization on an accelerator
                    # regardless by staging the LoRA parameters onto the compute device
                    # and moving the optimized parameters back afterwards.
                    storage_device = lora_A.device
                    compute_device = self._compute_device(storage_device)
                    lora_A.data = lora_A.data.to(compute_device)
                    lora_B.data = lora_B.data.to(compute_device)

                    # We need the base weight in float32 to compute the effective weight.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W_base = base_weight.to(
                            device=compute_device,
                            dtype=torch.float32,
                        )
                    else:
                        # Use the original dequantization logic from bitsandbytes.
                        W_base = BNB_F.dequantize_4bit(
                            base_weight.data,
                            quant_state,
                        ).to(
                            device=compute_device,
                            dtype=torch.float32,
                        )

                    # Pre-calculate the original row norms to preserve them.
                    # See https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                    W_row_norms = cast(
                        Tensor,
                        LA.vector_norm(W_base, dim=1, keepdim=True).detach(),
                    )

                    # Initialize A the same way PEFT does, but deterministically. See
                    # https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/lora/layer.py#L338
                    initial_A = torch.empty(lora_A.shape)
                    torch.nn.init.kaiming_uniform_(
                        initial_A,
                        a=math.sqrt(5),
                        generator=generator,
                    )
                    with torch.no_grad():
                        lora_A.copy_(initial_A)

                    # Move I/O tensors to the device of the adapter weights.
                    good_input, good_output = self.good_module_io[layer_index][
                        component
                    ][module_index]
                    bad_input, bad_output = self.bad_module_io[layer_index][component][
                        module_index
                    ]

                    good_input = good_input.float().to(lora_A.device)
                    good_output = good_output.float().to(lora_A.device)
                    bad_input = bad_input.float().to(lora_A.device)
                    bad_output = bad_output.float().to(lora_A.device)

                    def objective(A: Tensor, B: Tensor) -> Tensor:
                        # Calculate effective weight after applying adapter.
                        W_eff = W_base + (B @ A)

                        if self.settings.preserve_row_magnitudes:
                            # Normalize to unit length, then scale by original norms,
                            # preserving the original row norms.
                            W_eff = F.normalize(W_eff, p=2, dim=1) * W_row_norms

                        # Compute outputs using the effective weight.
                        new_good_output = good_input @ W_eff.T
                        new_bad_output = bad_input @ W_eff.T

                        return ara_loss(
                            good_output,
                            bad_output,
                            new_good_output,
                            new_bad_output,
                            parameters,
                        )

                    optimizer = LBFGS(
                        [lora_A, lora_B],
                        lr=self.settings.learning_rate,
                        max_iter=self.settings.max_iter,
                        history_size=self.settings.history_size,
                        line_search_fn="strong_wolfe",
                    )

                    def closure() -> Tensor:
                        optimizer.zero_grad()
                        loss = objective(lora_A, lora_B)
                        loss.backward()
                        return loss

                    for step in range(self.settings.n_optimization_steps):
                        loss = optimizer.step(closure)
                        if self.settings.print_loss:
                            print(
                                f"\\[{layer_index}/{component}/{module_index}] Step: {step + 1}, Loss: {loss.item():.6f}"
                            )

                    # Move the optimized parameters back to their storage device.
                    if compute_device != storage_device:
                        lora_A.data = lora_A.data.to(storage_device)
                        lora_B.data = lora_B.data.to(storage_device)

                    # Free the gradient buffers accumulated during optimization.
                    # Without this, they persist on the model (one full-size gradient
                    # per processed weight) and can easily consume tens of GB of VRAM,
                    # causing out-of-memory errors during the subsequent evaluation.
                    optimizer.zero_grad(set_to_none=True)

    def _compute_device(self, storage_device: torch.device) -> torch.device:
        """
        Device on which a per-module optimization runs.

        Uses the storage device when it is already an accelerator, so the default
        (GPU-resident model) path is unchanged, and otherwise falls back to CUDA so a
        parked, host-memory model still gets accelerated optimization.
        """
        if storage_device.type != "cpu":
            return storage_device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return storage_device

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.settings.lora_rank)
