# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Inference backend seam for scoring.

Scorers obtain responses through the `Context`. By default this goes through the
in-process HuggingFace `Model` (`HFBackend`). A `VLLMBackend` (see `vllm_backend.py`)
can serve the pure-inference evaluation path through vLLM instead.

Only the pure-inference scorer evaluation is routed here. Residuals, module I/O, and the
modifier optimization itself always stay on the HuggingFace model, because they need full
weights and per-module activations that vLLM does not expose.
"""

from typing import Protocol

import torch.nn.functional as F
from torch import Tensor

from .model import CaptureRequest, GenerationRecord, Model
from .utils import Prompt


class InferenceBackend(Protocol):
    def sync_adapter(self, model: Model) -> None:
        """
        Make the model's current modification (if any) visible to the backend.

        Called before each evaluation. Backends that read the model directly (the HF
        backend) do nothing; a backend that keeps its own engine copies the model's
        current adapter over.
        """
        ...

    def get_generation_records(
        self, prompts: list[Prompt], *, capture: CaptureRequest
    ) -> list[GenerationRecord]: ...

    def get_logprobs_prompt(self, prompts: list[Prompt]) -> Tensor:
        """
        Log-probability of every vocabulary token at the end of the prompt (i.e. for the
        first response token).
        """
        ...

    def get_logprobs_from_capture_ids(self, capture_ids: list[list[int]]) -> Tensor:
        """
        Log-probability of every vocabulary token at the position following each given
        token sequence.
        """
        ...

    def pause(self) -> None:
        """
        Release GPU memory the backend holds, so the modifier can use the GPU during the
        per-trial modification. No-op for backends that hold no GPU memory.
        """
        ...

    def resume(self) -> None:
        """Restore the backend after the modifier has run. Inverse of `pause`."""
        ...


class HFBackend:
    """Default backend: generate with the in-process HuggingFace model."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def sync_adapter(self, model: Model) -> None:
        return

    def get_generation_records(
        self, prompts: list[Prompt], *, capture: CaptureRequest
    ) -> list[GenerationRecord]:
        return self._model.get_response_records_batched(
            prompts,
            skip_special_tokens=True,
            capture=capture,
        )

    def get_logprobs_prompt(self, prompts: list[Prompt]) -> Tensor:
        logits = self._model.get_logits_batched(prompts, after_reasoning=False)
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_from_capture_ids(self, capture_ids: list[list[int]]) -> Tensor:
        logits = self._model.get_logits_from_capture_ids(capture_ids)
        return F.log_softmax(logits, dim=-1)

    def pause(self) -> None:
        return

    def resume(self) -> None:
        return
