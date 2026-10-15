# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
vLLM-backed scorer evaluation.

`VLLMBackend` serves the pure-inference scorer evaluation from a vLLM engine, applying
the modifier's current LoRA adapter per trial. It reuses the HuggingFace model only for
tokenization, prompt rendering, and locating capture positions in the generated token
IDs; generation itself happens in vLLM.

First-milestone scope: text-only scorers. Requests to capture logits or module I/O are
served with empty captures, so a scorer that needs them would fall back to the
HuggingFace capture path rather than fail; the evaluator only selects this backend when
every configured scorer is text-only.
"""

import atexit
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import CaptureRequest, GenerationRecord, Model
from .utils import Prompt, print


def target_module_suffixes(names: list[str]) -> list[str]:
    """
    Reduce LoRA target module names to their last dotted component.

    Heretic records fully-qualified module names such as
    "model.layers.0.self_attn.o_proj"; vLLM binds adapters by module suffix such as
    "o_proj".
    """
    return sorted({name.rsplit(".", 1)[-1] for name in names})


def normalize_adapter_target_modules(config_path: Path) -> None:
    """
    Rewrite an exported PEFT adapter config so its `target_modules` are suffixes.

    Leaves the file untouched when `target_modules` is absent or not a list.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = config.get("target_modules")
    if not isinstance(targets, list):
        return
    config["target_modules"] = target_module_suffixes(
        [str(target) for target in targets]
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


class VLLMBackend:
    """
    Inference backend that serves scorer generation from a vLLM engine.

    The engine is built lazily on first use, so importing this module does not require
    vLLM unless the backend is actually enabled and used.
    """

    def __init__(self, settings: Settings, model: Model) -> None:
        self._settings = settings
        self._vllm = settings.vllm
        self._model = model
        self._engine: Any = None
        self._vocab_size: int | None = None
        self._adapter_dir = Path(tempfile.mkdtemp(prefix="heretic-vllm-adapters-"))
        self._adapter_request: Any = None
        self._adapter_counter = 0
        atexit.register(self.close)

    def close(self) -> None:
        """Release the engine and remove exported adapter files."""
        self._engine = None
        self._adapter_request = None
        shutil.rmtree(self._adapter_dir, ignore_errors=True)

    def sync_adapter(self, model: Model) -> None:
        """
        Export the model's current LoRA adapter (if any) and register it with vLLM.

        vLLM binds adapters by module suffix, so the exported config's fully-qualified
        `target_modules` are reduced to suffixes. When the model carries no adapter (the
        unmodified base), no adapter is applied.
        """
        from peft import PeftModel

        peft_model = model.model
        if not isinstance(peft_model, PeftModel):
            self._adapter_request = None
            return

        self._adapter_counter += 1
        adapter_dir = self._adapter_dir / f"adapter-{self._adapter_counter}"
        peft_model.save_pretrained(str(adapter_dir))
        normalize_adapter_target_modules(adapter_dir / "adapter_config.json")

        from vllm.lora.request import LoRARequest  # ty: ignore[unresolved-import]

        self._adapter_request = LoRARequest(
            lora_name=f"heretic-{self._adapter_counter}",
            lora_int_id=self._adapter_counter,
            lora_path=str(adapter_dir),
        )

    def get_generation_records(
        self, prompts: list[Prompt], *, capture: CaptureRequest
    ) -> list[GenerationRecord]:
        from vllm import SamplingParams  # ty: ignore[unresolved-import]
        from vllm.inputs import TokensPrompt  # ty: ignore[unresolved-import]

        engine = self._get_engine()
        tokenizer = self._model.tokenizer

        prompt_ids = [
            tokenizer(chat_prompt)["input_ids"]
            for chat_prompt in self._model.build_chat_prompts(prompts)
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self._model.response_max_new_tokens(),
        )
        inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]

        outputs = engine.generate(
            inputs,
            sampling_params,
            lora_request=self._adapter_request,
        )

        records: list[GenerationRecord] = []
        for ids, output in zip(prompt_ids, outputs):
            generated_ids = list(output.outputs[0].token_ids)
            answer_ids, capture_ids, unfinished = self._model.answer_and_capture_ids(
                ids, generated_ids
            )
            records.append(
                GenerationRecord(
                    response=cast(
                        str, tokenizer.decode(answer_ids, skip_special_tokens=True)
                    ),
                    capture_ids=capture_ids,
                    first_content_ids=self._model.first_content_ids(ids, generated_ids),
                    unfinished=unfinished,
                    repetition_aborted=False,
                    generated_token_count=len(generated_ids),
                )
            )
        return records

    def get_logprobs_prompt(self, prompts: list[Prompt]) -> Tensor:
        tokenizer = self._model.tokenizer
        prompt_ids = [
            tokenizer(chat_prompt)["input_ids"]
            for chat_prompt in self._model.build_chat_prompts(prompts)
        ]
        return self.get_logprobs_from_capture_ids(prompt_ids)

    def get_logprobs_from_capture_ids(self, capture_ids: list[list[int]]) -> Tensor:
        """
        Full-vocabulary log-probabilities at the position after each token sequence.

        Runs one generation-free forward per sequence (a single token with full-vocab
        logprobs), so only the capture position is materialized instead of a full
        distribution at every generated token.
        """
        from vllm import SamplingParams  # ty: ignore[unresolved-import]
        from vllm.inputs import TokensPrompt  # ty: ignore[unresolved-import]

        engine = self._get_engine()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=-1)
        inputs = [TokensPrompt(prompt_token_ids=ids) for ids in capture_ids]
        outputs = engine.generate(
            inputs,
            sampling_params,
            lora_request=self._adapter_request,
        )

        vocab_size = self._get_vocab_size()
        rows: list[Tensor] = []
        for output in outputs:
            entry = output.outputs[0].logprobs[0]
            row = torch.full((vocab_size,), float("-inf"))
            token_ids = torch.tensor(list(entry.keys()), dtype=torch.long)
            row[token_ids] = torch.tensor([value.logprob for value in entry.values()])
            rows.append(row)

        tensor = torch.stack(rows, dim=0)
        if self._vllm.logprobs_mode == "raw_logits":
            tensor = F.log_softmax(tensor, dim=-1)
        elif self._vllm.logprobs_mode != "raw_logprobs":
            raise NotImplementedError(
                "The vLLM KL path supports logprobs_mode 'raw_logprobs' and "
                f"'raw_logits', not {self._vllm.logprobs_mode!r}."
            )
        return tensor.to(self._model.output_device())

    def _get_vocab_size(self) -> int:
        if self._vocab_size is None:
            engine = self._get_engine()
            model_config = engine.llm_engine.model_config
            self._vocab_size = int(model_config.get_vocab_size())
        return self._vocab_size

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        if self._vllm.mode != "inprocess":
            raise NotImplementedError(
                "vLLM server mode is not implemented yet; set mode = 'inprocess'."
            )

        from vllm import LLM  # ty: ignore[unresolved-import]

        model = self._vllm.model or self._settings.model
        print(f"Loading vLLM engine for [bold]{model}[/]...")

        self._engine = LLM(
            model=model,
            dtype=self._vllm.dtype,
            tensor_parallel_size=self._vllm.tensor_parallel_size,
            gpu_memory_utilization=self._vllm.gpu_memory_utilization,
            max_model_len=self._vllm.max_model_len or None,
            enable_lora=True,
            max_lora_rank=self._vllm.max_lora_rank,
            max_logprobs=self._vllm.max_logprobs,
            logprobs_mode=self._vllm.logprobs_mode,
            enforce_eager=self._vllm.enforce_eager,
            trust_remote_code=self._vllm.trust_remote_code,
        )
        return self._engine
