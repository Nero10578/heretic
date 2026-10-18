# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeAlias, cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available

from .config import QuantizationMethod, Settings
from .system import empty_cache
from .utils import Prompt, batchify, format_exception, print


def get_model_class(
    model: str,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    configs = PretrainedConfig.get_config_dict(model)

    if any(("vision_config" in config) for config in configs):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


# Some chat templates (e.g. gpt-oss) insert the current date into the prompt,
# which makes the residuals, and therefore the modified model, depend on the day
# Heretic is run. Rendering templates with a fixed date keeps results reproducible.
def strftime_fixed_date(format: str) -> str:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(format)


# The list contains one element per layer.
# Each element maps from the component name to a (possibly sparse) mapping
# from the module index to an (input, output) tuple containing the I/O
# tensors of shape (prompt, component).
ModuleIO: TypeAlias = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]


@dataclass
class GenerationRecord:
    """
    Result of generating a response for a single prompt.

    Besides the decoded answer, it retains the token IDs up to two capture positions:
    the first answer token (after the reasoning block closes), and the first token of
    the response that is not part of the model's thinking-start marker. Keeping these in
    one record lets response decoding (refusal/keyword scoring) and logit or residual
    capture share a single generation instead of regenerating the reasoning for each.
    """

    response: str
    capture_ids: list[int]
    first_content_ids: list[int]
    unfinished: bool
    repetition_aborted: bool
    generated_token_count: int
    # Captured during generation (only when capture_during_generation is enabled and the
    # reasoning block closed, see _GenerationCapture). None means the value must be
    # obtained with a separate forward over prompt + reasoning.
    answer_logits: Tensor | None = None
    content_logits: Tensor | None = None
    module_io_row: ModuleIO | None = None


def reasoning_start_skip_length(
    prompt_ids: list[int],
    generated_ids: list[int],
    markers: list[str],
    decode: Callable[[list[int]], str],
    special_token_ids: set[int],
    probe_tokens: int = 8,
    window_tokens: int = 128,
) -> int:
    """
    Number of leading generated tokens to skip to reach the first token that carries
    content, rather than part of the model's thinking-start marker (e.g. the
    `<|channel>thought` opener).

    The marker is found by decoding the prompt plus growing prefixes of the generation
    and checking whether the text ends with one of the configured markers. Decoding
    avoids depending on how the model happens to split the marker into tokens, or
    whether those tokens are special, and it also handles a marker that the template
    already opened. If no marker is found, leading special tokens are skipped.
    """
    for skip in range(1, min(len(generated_ids), probe_tokens) + 1):
        text = decode((prompt_ids + generated_ids[:skip])[-window_tokens:])
        if any(text.endswith(marker) for marker in markers):
            return skip

    skip = 0
    while skip < len(generated_ids) and generated_ids[skip] in special_token_ids:
        skip += 1
    return skip


class RepetitionLoopError(Exception):
    """
    Raised when a generation is so dominated by repetition loops that the trial cannot be
    scored meaningfully. The optimization loop prunes the trial in response.
    """


class RepetitionStoppingLogitsProcessor(LogitsProcessor):
    """
    Forces end-of-sequence for any row whose generated tokens have fallen into a
    repetition loop, so a broken model stops instead of emitting the full token budget
    of gibberish. The prompt length is subtracted out so the prompt text is ignored.
    """

    def __init__(
        self,
        prompt_length: int,
        eos_token_ids: list[int],
        pad_token_id: int | None,
        period_max: int,
        min_repeats: int,
    ) -> None:
        self._prompt_length = prompt_length
        self._eos_token_ids = eos_token_ids
        self._pad_token_id = pad_token_id
        self._period_max = period_max
        self._min_repeats = min_repeats
        self._aborted_rows: set[int] = set()

    @property
    def aborted_rows(self) -> set[int]:
        return self._aborted_rows

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        generated = input_ids[:, self._prompt_length :]
        generated_length = generated.shape[1]
        if generated_length <= 0 or not self._eos_token_ids:
            return scores

        # Rows that already finished are padded with the pad token after their EOS, so
        # their tails are full of repeated pad tokens. Those must not count as loops.
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        for eos_token_id in self._eos_token_ids:
            finished |= (generated == eos_token_id).any(dim=1)
        if self._pad_token_id is not None:
            finished |= (generated == self._pad_token_id).any(dim=1)

        repetitive = torch.zeros_like(finished)
        for period in range(1, self._period_max + 1):
            needed = period * self._min_repeats
            if needed > generated_length:
                continue
            tail = generated[:, -needed:]
            repetitive |= (tail[:, period:] == tail[:, :-period]).all(dim=1)
        repetitive &= ~finished

        if repetitive.any():
            self._aborted_rows.update(
                repetitive.nonzero(as_tuple=False).flatten().tolist()
            )
            for eos_token_id in self._eos_token_ids:
                scores[repetitive, eos_token_id] = float("inf")

        return scores


@dataclass
class CaptureRequest:
    """
    Which captures to take while generating a response. Model.get_response_records only
    acts on this when `capture_during_generation` is enabled and the model supports it
    (reasoning mode with a configured closing tag); otherwise the records are returned
    uncaptured and callers fall back to a separate forward.
    """

    logits: bool = False
    module_io: bool = False


def merge_module_io_rows(rows: list[ModuleIO], layer_count: int) -> ModuleIO:
    """
    Combine per-prompt module I/O (each with one row per module) into a single ModuleIO,
    concatenating every module's rows in prompt order. Mirrors get_module_io_batched.
    """
    merged: ModuleIO = [{} for _ in range(layer_count)]
    for row in rows:
        for layer_index, components in enumerate(row):
            for component, modules in components.items():
                target = merged[layer_index].setdefault(component, {})
                for module_index, (input, output) in modules.items():
                    existing = target.get(module_index)
                    if existing is None:
                        target[module_index] = (input, output)
                    else:
                        target[module_index] = (
                            torch.cat([existing[0], input], dim=0),
                            torch.cat([existing[1], output], dim=0),
                        )
    return merged


def split_module_io_rows(module_io: ModuleIO, count: int) -> list[ModuleIO]:
    """
    Split a ModuleIO whose module tensors hold `count` rows into one single-row ModuleIO
    per prompt. Only valid when every row is present for every module, which holds for
    dense models (the only case where captured and fallback rows are mixed).
    """
    rows: list[ModuleIO] = [[{} for _ in module_io] for _ in range(count)]
    for layer_index, components in enumerate(module_io):
        for component, modules in components.items():
            for module_index, (input, output) in modules.items():
                if input.shape[0] != count:
                    raise ValueError(
                        "Module I/O rows do not match the expected prompt count"
                    )
                for row_index in range(count):
                    rows[row_index][layer_index].setdefault(component, {})[
                        module_index
                    ] = (
                        input[row_index : row_index + 1],
                        output[row_index : row_index + 1],
                    )
    return rows


class _GenerationCapture:
    """
    Captures logits and/or module I/O at the capture positions while a response is being
    generated, so no separate forward over prompt + reasoning is needed.

    Positions are located online from the tokens the generator produces:
    - the first-content position, where the decoded generation first ends with a
      thinking-start marker (probed within a short window),
    - the answer position, where the generated tokens first end with a
      chain-of-thought closing tag, advanced past the whitespace separator that chat
      templates place before the answer (e.g. Qwen3's "\n\n") to the first token that
      carries answer content.

    The distribution a logits processor receives at the step that appended token index
    `a` is exactly the distribution a forward over `prompt + generated[:a]` would return,
    and the module I/O captured in that same forward is the I/O at its last position, so
    the online captures match the separate capture forward. Rows whose reasoning block
    never closes (or whose marker/tag is not found online) stay uncaptured and the caller
    falls back to the separate forward for them.
    """

    def __init__(
        self,
        model: "Model",
        prompt_length: int,
        prompt_ids: list[list[int]],
        capture_logits: bool,
        capture_module_io: bool,
    ) -> None:
        self._model = model
        self._prompt_length = prompt_length
        self._prompt_ids = prompt_ids
        self._capture_logits = capture_logits
        self._capture_module_io = capture_module_io

        self._close_tag_ids = model._reasoning_close_tag_ids()
        self._max_tag_length = max((len(tag) for tag in self._close_tag_ids), default=0)
        self._start_markers = model._reasoning_start_markers()
        self._probe_tokens = 8
        self._window_tokens = 128

        self.answer_logits: dict[int, Tensor] = {}
        self.content_logits: dict[int, Tensor] = {}
        self.module_io_rows: dict[int, ModuleIO] = {}
        # Generated-token index (appended count) at which each capture was taken, used to
        # verify against the post-hoc capture position before trusting the capture.
        self.answer_positions: dict[int, int] = {}
        self.content_positions: dict[int, int] = {}
        self._answer_done: set[int] = set()

        # Rows whose reasoning block has closed and whose leading whitespace separator
        # (e.g. Qwen3's "\n\n" before the answer) is still being skipped. The answer
        # capture is deferred until the first token that carries content appears, using
        # the scores and module I/O from the preceding step.
        self._separator_rows: set[int] = set()
        self._pending_answer_scores: dict[int, Tensor] = {}
        self._pending_answer_positions: dict[int, int] = {}

        # Most recent forward's per-module I/O, overwritten every step and cleared at the
        # start of each forward so that modules that did not fire this step are absent.
        # The previous step's I/O is retained because the answer anchor can be one step
        # after the closing tag when the template emits a whitespace separator.
        self._io: dict[int, dict[str, dict[int, tuple[Tensor, Tensor]]]] = {}
        self._previous_io: dict[int, dict[str, dict[int, tuple[Tensor, Tensor]]]] = {}
        self._io_complete = False
        self._previous_io_complete = False
        self._expected_io_count = 0

        self._layer_count = len(model.get_layers())
        self._handles: list[RemovableHandle] = []

    def register(self) -> None:
        self._handles.append(
            self._model.model.register_forward_pre_hook(self._clear_io)
        )

        if not self._capture_module_io:
            return

        for layer_index in range(self._layer_count):
            for component, modules in self._model.get_layer_modules(
                layer_index
            ).items():
                self._expected_io_count += len(modules)
                for module_index, module in enumerate(modules):
                    self._handles.append(
                        module.register_forward_hook(
                            self._make_hook(layer_index, component, module_index)
                        )
                    )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _clear_io(self, module: Module, args: Any) -> None:
        # Reassign rather than clear: the previous step's I/O is retained so the answer
        # can still be anchored when a whitespace separator follows the closing tag.
        self._io = {}

    def _make_hook(
        self,
        layer_index: int,
        component: str,
        module_index: int,
    ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
        def hook(
            module: Module,
            inputs: tuple[Tensor, ...],
            outputs: Tensor,
        ) -> None:
            components = self._io.setdefault(layer_index, {}).setdefault(component, {})
            # At prefill the module input/output span the whole prompt, which can be very
            # large; keep only the last position so the full activation is not retained.
            # At decode steps (one position) a view is enough.
            if inputs[0].shape[1] > 1:
                components[module_index] = (
                    inputs[0][:, -1, :].detach().clone(),
                    outputs[:, -1, :].detach().clone(),
                )
            else:
                components[module_index] = (inputs[0][:, -1, :], outputs[:, -1, :])

        return hook

    def processor(self) -> LogitsProcessor:
        return _CaptureLogitsProcessor(self)

    def _store_logits(self, row_logits: Tensor) -> Tensor:
        return row_logits.detach().to(self._model.output_device())

    def _build_row(
        self,
        row_index: int,
        io: dict[int, dict[str, dict[int, tuple[Tensor, Tensor]]]],
    ) -> ModuleIO:
        row: ModuleIO = [{} for _ in range(self._layer_count)]
        for layer_index, components in io.items():
            for component, modules in components.items():
                for module_index, (input, output) in modules.items():
                    row[layer_index].setdefault(component, {})[module_index] = (
                        input[row_index : row_index + 1].detach().clone().cpu(),
                        output[row_index : row_index + 1].detach().clone().cpu(),
                    )
        return row

    def observe(self, input_ids: Tensor, scores: Tensor) -> None:
        generated = input_ids[:, self._prompt_length :]
        appended = generated.shape[1]
        if appended <= 0:
            return

        if self._capture_module_io:
            present = sum(
                len(modules)
                for components in self._io.values()
                for modules in components.values()
            )
            self._io_complete = present == self._expected_io_count
        else:
            self._io_complete = False

        if self._max_tag_length:
            tail = generated[:, -self._max_tag_length :].tolist()
        else:
            tail = [[] for _ in range(input_ids.shape[0])]

        last_tokens = generated[:, -1].tolist()

        for row_index in range(input_ids.shape[0]):
            content_done = row_index in self.content_logits

            if row_index not in self._answer_done:
                if row_index in self._separator_rows:
                    # The reasoning block has closed. Skip the template's whitespace
                    # separator and anchor the answer at the first token that carries
                    # content, using the distribution from the preceding step (which is
                    # the one that predicted this token).
                    if self._model.decode_tokens([last_tokens[row_index]]).strip():
                        self._capture_answer(
                            row_index,
                            self._pending_answer_positions[row_index],
                            self._pending_answer_scores[row_index],
                            self._previous_io,
                            self._previous_io_complete,
                        )
                        self._separator_rows.discard(row_index)
                elif self._max_tag_length and any(
                    len(tail[row_index]) >= len(tag)
                    and tail[row_index][-len(tag) :] == tag
                    for tag in self._close_tag_ids
                ):
                    self._separator_rows.add(row_index)

            if (
                self._capture_logits
                and not content_done
                and appended <= self._probe_tokens
                and self._start_markers
            ):
                ids = self._prompt_ids[row_index] + generated[row_index].tolist()
                text = self._model.decode_tokens(ids[-self._window_tokens :])
                if any(text.endswith(marker) for marker in self._start_markers):
                    self.content_logits[row_index] = self._store_logits(
                        scores[row_index]
                    )
                    self.content_positions[row_index] = appended

        for row_index in self._separator_rows:
            self._pending_answer_scores[row_index] = scores[row_index]
            self._pending_answer_positions[row_index] = appended

        self._previous_io = self._io
        self._previous_io_complete = self._io_complete

    def _capture_answer(
        self,
        row_index: int,
        position: int,
        row_scores: Tensor,
        io: dict[int, dict[str, dict[int, tuple[Tensor, Tensor]]]],
        io_complete: bool,
    ) -> None:
        if self._capture_logits:
            self.answer_logits[row_index] = self._store_logits(row_scores)
        if self._capture_module_io and io_complete:
            self.module_io_rows[row_index] = self._build_row(row_index, io)
        self.answer_positions[row_index] = position
        self._answer_done.add(row_index)


class _CaptureLogitsProcessor(LogitsProcessor):
    """
    Drives _GenerationCapture from inside the generation loop. Registered before the
    repetition processor so it sees the raw (unmodified) scores.
    """

    def __init__(self, capture: _GenerationCapture) -> None:
        self._capture = capture

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        self._capture.observe(input_ids, scores)
        return scores


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False
        self._cot_close_tag_ids_cache: list[list[int]] | None = None
        self._special_token_ids_cache: set[int] | None = None
        self._unfinished_reasoning_dumped = False

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        # PreTrainedTokenizerBase is the "base class for all tokenizer backends"
        # according to the documentation.
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            ),
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(settings.model) == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        self.trusted_models = set()

        loaded = False

        for dtype in settings.dtypes:
            if loaded:
                break

            for attention in self._attention_candidates(settings.attn_implementation):
                if attention is None:
                    print(f"* Trying dtype [bold]{dtype}[/]...")
                else:
                    print(
                        f"* Trying dtype [bold]{dtype}[/] with "
                        f"[bold]{attention}[/] attention..."
                    )

                try:
                    quantization_config = self._get_quantization_config(dtype)

                    extra_kwargs = {}
                    # Only include quantization_config if it's not None
                    # (some models like gpt-oss have issues with explicit None).
                    if quantization_config is not None:
                        extra_kwargs["quantization_config"] = quantization_config
                    if attention is not None:
                        extra_kwargs["attn_implementation"] = attention

                    self.model = get_model_class(settings.model).from_pretrained(
                        settings.model,
                        dtype=dtype,
                        device_map=settings.device_map,
                        max_memory=self.max_memory,
                        trust_remote_code=True
                        if settings.model in self.trusted_models
                        else None,
                        **self.revision_kwargs,
                        **extra_kwargs,
                    )

                    self.dtype = self.model.dtype

                    # If we reach this point and the model requires trust_remote_code,
                    # the user must have agreed when prompted to execute remote code,
                    # because from_pretrained raises an exception otherwise.
                    self.trusted_models.add(settings.model)

                    # A test run can reveal dtype-related problems such as the infamous
                    # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                    # (https://github.com/meta-llama/llama/issues/380).
                    self.generate(
                        [
                            Prompt(
                                system=settings.system_prompt,
                                user="What is 1+1?",
                            )
                        ],
                        max_new_tokens=1,
                    )
                except Exception as error:
                    self.model = None
                    empty_cache()

                    formatted = format_exception(error)
                    if "\n" in formatted:
                        print(f"* [red]Failed:\n{formatted}[/]")
                    else:
                        print(f"* [red]Failed ({formatted})[/]")

                    continue

                if settings.quantization == QuantizationMethod.BNB_4BIT:
                    print("* Quantized to 4-bit precision")

                if attention is not None:
                    print(f"* Using [bold]{attention}[/] for attention")

                loaded = True
                break

        if self.model is None:
            raise Exception(
                "Failed to load model with all configured dtypes and attention implementations."
            )

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Abliterable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def apply_lora(self, lora_rank: int):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for abliteration (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        # PEFT initializes A randomly from the global RNG. B is zero, so this doesn't
        # change the model, but zeroing A too keeps saved adapters reproducible.
        self._zero_lora_weights()

    def _zero_lora_weights(self):
        for name, module in self.model.named_modules():
            if ("lora_A" in name or "lora_B" in name) and hasattr(module, "weight"):
                torch.nn.init.zeros_(module.weight)

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        return None

    @staticmethod
    def _attention_candidates(requested: str) -> list[str | None]:
        """
        Attention implementations to try, in order.

        "auto" prefers FlashAttention when the flash-attn package is installed (it is
        much faster for the long reasoning sequences) and otherwise leaves the choice to
        transformers, which defaults to SDPA. An explicit request is tried first, then
        SDPA, then the library default, so an unsupported request degrades gracefully.
        """
        if requested == "auto":
            if is_flash_attn_2_available():
                return ["flash_attention_2", "sdpa"]
            if is_flash_attn_3_available():
                return ["flash_attention_3", "sdpa"]
            return [None]

        if requested == "sdpa":
            return ["sdpa"]

        return [requested, "sdpa", None]

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        # Check if we need special handling for quantized models
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = get_peft_model(base_model, self.peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self) -> bool:
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.

        Returns True if the fast path was taken.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            # Zeroing A as well ensures that adapter weights of modules not modified
            # by the next trial don't depend on what previous trials did.
            self._zero_lora_weights()
            return True

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )

        self.needs_reload = False

        return False

    def park(self) -> None:
        """
        Move all model parameters and buffers to CPU memory to free GPU memory.

        Used to hand the GPU to another engine (e.g. vLLM) after the model's one-time
        initialization. The model is not usable for forwards while parked; callers stage
        the individual tensors they need back onto an accelerator themselves.
        """
        empty_cache()
        for parameter in self.model.parameters():
            parameter.data = parameter.data.to("cpu")
        for buffer in self.model.buffers():
            buffer.data = buffer.data.to("cpu")
        empty_cache()

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

        # Text-only models.
        return model.model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

    def output_device(self) -> torch.device:
        """
        Device on which captured logits/residuals are returned. Matches the
        offload_outputs_to_cpu convention used by the capture forwards.
        """
        if self.settings.offload_outputs_to_cpu:
            return torch.device("cpu")
        return self.model.device

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty: ignore[unresolved-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)  # ty: ignore[unresolved-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty: ignore[unresolved-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty: ignore[unresolved-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.output_linear)  # ty: ignore[unresolved-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    def get_abliterable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def _build_chat_prompts(self, prompts: list[Prompt]) -> list[str]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
                # Overrides the function that templates call to get the current date.
                strftime_now=strftime_fixed_date,
                # Extra template options (e.g. enable_thinking) from the config.
                **self.settings.chat_template_kwargs,
            ),
        )

        if (
            self.settings.response_prefix
            and not self.settings.generate_chain_of_thought
        ):
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        return chat_prompts

    def _tokenize_prompts(self, prompts: list[Prompt]) -> BatchEncoding:
        return self.tokenizer(
            self._build_chat_prompts(prompts),
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

    def _generate_from_inputs(
        self,
        inputs: BatchEncoding,
        **kwargs: Any,
    ) -> GenerateDecoderOnlyOutput | LongTensor:
        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        return self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        inputs = self._tokenize_prompts(prompts)
        outputs = self._generate_from_inputs(inputs, **kwargs)
        return inputs, outputs

    def _capture_inputs(
        self,
        prompts: list[Prompt],
        after_reasoning: bool | None = None,
    ) -> BatchEncoding:
        """
        Tokenized inputs at the position where residuals, module I/O, and logits should
        be captured. Normally this is the last prompt position (i.e. the first response
        token). If `after_reasoning` is true, it is instead the first answer token, after
        the model's own reasoning. When `after_reasoning` is None, the global
        generate_chain_of_thought setting decides.
        """
        if after_reasoning is None:
            after_reasoning = self.settings.generate_chain_of_thought
        if after_reasoning:
            return self._answer_start_inputs(prompts)
        return self._tokenize_prompts(prompts)

    def _answer_start_inputs(self, prompts: list[Prompt]) -> BatchEncoding:
        """
        Tokenized inputs positioned at the first answer token: the chat prompt followed
        by the model's own generated reasoning up to and including the chain-of-thought
        closing tag. This ensures the ablation is aimed at, and KL divergence is measured
        at, the point where the model actually answers. Prompts whose reasoning does not
        finish within the token budget fall back to the full generation.
        """
        records = self.get_response_records(prompts)
        return self._capture_inputs_from_ids([record.capture_ids for record in records])

    def _capture_inputs_from_ids(self, capture_ids: list[list[int]]) -> BatchEncoding:
        """
        Left-pad the given per-prompt capture token IDs into one batch, so the capture
        position is the last position of every row. Padding is masked out, and the
        prompt padding that generation introduced is already stripped, so the result
        does not depend on which batch the tokens were generated in.
        """
        pad_id = self.tokenizer.pad_token_id

        max_length = max(len(ids) for ids in capture_ids)
        input_ids = torch.full((len(capture_ids), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(capture_ids), max_length), dtype=torch.long)
        for index, ids in enumerate(capture_ids):
            length = len(ids)
            input_ids[index, max_length - length :] = torch.tensor(
                ids, dtype=torch.long
            )
            attention_mask[index, max_length - length :] = 1

        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        ).to(self.model.device)

    def _response_max_new_tokens(self) -> int:
        """
        Maximum number of tokens a full response generation may produce. In reasoning
        mode this covers the reasoning phase plus the final answer.
        """
        if self.settings.generate_chain_of_thought:
            return self.settings.max_thought_length + self.settings.max_response_length
        return self.settings.max_response_length

    def _capture_during_generation_supported(self) -> bool:
        """
        Whether captures can be taken while generating. Requires the feature to be
        enabled in reasoning mode with a configured chain-of-thought closing tag, so
        that the answer position can be located from the generated tokens.
        """
        return (
            self.settings.capture_during_generation
            and self.settings.generate_chain_of_thought
            and bool(self._reasoning_close_tag_ids())
        )

    def get_response_records(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        capture: CaptureRequest | None = None,
    ) -> list[GenerationRecord]:
        """
        Generate one response per prompt and return, for each, the decoded answer
        together with the token IDs up to the answer-capture position. Sharing these
        lets refusal scoring and logit/residual capture reuse a single generation
        instead of regenerating the reasoning separately.

        When `capture` requests logits and/or module I/O and `capture_during_generation`
        is enabled, they are taken during this same generation (see _GenerationCapture),
        so no separate forward over prompt + reasoning is needed.
        """
        inputs = self._tokenize_prompts(prompts)
        input_ids = cast(Tensor, inputs["input_ids"])
        attention_mask = cast(Tensor, inputs["attention_mask"])
        prompt_length = input_ids.shape[1]
        # Real (unpadded) prompt token IDs per row, needed to locate the marker position
        # online during generation. The prompt is left-padded (see __init__).
        prompt_ids_per_row = [
            input_ids[index][attention_mask[index] == 1].tolist()
            for index in range(input_ids.shape[0])
        ]

        processors: list[LogitsProcessor] = []
        capture_state: _GenerationCapture | None = None
        if (
            capture is not None
            and (capture.logits or capture.module_io)
            and self._capture_during_generation_supported()
        ):
            capture_state = _GenerationCapture(
                model=self,
                prompt_length=prompt_length,
                prompt_ids=prompt_ids_per_row,
                capture_logits=capture.logits,
                capture_module_io=capture.module_io,
            )
            capture_state.register()
            # First so it sees the raw scores, before the repetition processor edits them.
            processors.append(capture_state.processor())

        repetition_processor = None
        if self.settings.abort_repeating_generation:
            repetition_processor = RepetitionStoppingLogitsProcessor(
                prompt_length=prompt_length,
                eos_token_ids=self._eos_token_ids(),
                pad_token_id=self.tokenizer.pad_token_id,
                period_max=self.settings.repetition_period_max,
                min_repeats=self.settings.repetition_min_repeats,
            )
            processors.append(repetition_processor)

        extra_kwargs: dict[str, Any] = {}
        if processors:
            extra_kwargs["logits_processor"] = LogitsProcessorList(processors)

        try:
            outputs = self._generate_from_inputs(
                inputs,
                max_new_tokens=self._response_max_new_tokens(),
                **extra_kwargs,
            )
        finally:
            if capture_state is not None:
                capture_state.remove()

        generated = outputs[:, prompt_length:]
        pad_id = self.tokenizer.pad_token_id

        repetition_aborted_rows = (
            repetition_processor.aborted_rows
            if repetition_processor is not None
            else set()
        )
        records: list[GenerationRecord] = []

        for index, row in enumerate(generated.tolist()):
            prompt_ids = prompt_ids_per_row[index]

            # Strip any trailing padding left after a sequence hit EOS.
            while row and row[-1] == pad_id:
                row.pop()

            if self.settings.generate_chain_of_thought:
                answer_ids, capture_ids, unfinished = self._answer_and_capture_ids(
                    prompt_ids, row
                )
                first_content_ids = self._first_content_ids(prompt_ids, row)
            else:
                answer_ids = row[: self.settings.max_response_length]
                capture_ids = prompt_ids + answer_ids
                first_content_ids = prompt_ids
                unfinished = False

            if (
                unfinished
                and self.settings.print_debug_information
                and not self._unfinished_reasoning_dumped
            ):
                # One-shot dump to diagnose why reasoning never closed.
                self._unfinished_reasoning_dumped = True
                prompt_text = self._build_chat_prompts([prompts[index]])[0]
                raw = cast(
                    str,
                    self.tokenizer.decode(row, skip_special_tokens=False),
                )
                print()
                print("Debug: reasoning never closed (one-shot dump).", markup=False)
                print(
                    f"* Expected closing tag ids: {self._reasoning_close_tag_ids()}",
                    markup=False,
                )
                print(f"* Prompt tail: {prompt_text[-200:]!r}", markup=False)
                print(f"* {len(row)} tokens generated", markup=False)
                print(f"* Raw head: {raw[:300]!r}", markup=False)
                print(f"* Raw tail: {raw[-400:]!r}", markup=False)

            answer_logits: Tensor | None = None
            content_logits: Tensor | None = None
            module_io_row: ModuleIO | None = None
            if capture_state is not None:
                prompt_token_count = len(prompt_ids)
                # Only trust a capture when the online-detected position matches the
                # post-hoc capture position, so a mismatch cannot silently feed the
                # wrong distribution or module I/O into evaluation.
                if (
                    capture_state.answer_positions.get(index)
                    == len(capture_ids) - prompt_token_count
                ):
                    answer_logits = capture_state.answer_logits.get(index)
                    module_io_row = capture_state.module_io_rows.get(index)
                if (
                    capture_state.content_positions.get(index)
                    == len(first_content_ids) - prompt_token_count
                ):
                    content_logits = capture_state.content_logits.get(index)

            records.append(
                GenerationRecord(
                    response=cast(
                        str,
                        self.tokenizer.decode(
                            answer_ids,
                            skip_special_tokens=skip_special_tokens,
                        ),
                    ),
                    capture_ids=capture_ids,
                    first_content_ids=first_content_ids,
                    unfinished=unfinished,
                    repetition_aborted=index in repetition_aborted_rows,
                    generated_token_count=len(row),
                    answer_logits=answer_logits,
                    content_logits=content_logits,
                    module_io_row=module_io_row,
                )
            )

        return records

    def get_responses_with_status(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> tuple[list[str], int, int]:
        """
        Like get_responses, but also returns the number of responses whose answer is
        empty because the reasoning block never closed (only nonzero in reasoning mode),
        and the largest number of tokens generated for any prompt in this call.
        """
        records = self.get_response_records(
            prompts,
            skip_special_tokens=skip_special_tokens,
        )
        return (
            [record.response for record in records],
            sum(record.unfinished for record in records),
            max(
                (record.generated_token_count for record in records),
                default=0,
            ),
        )

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [
            record.response
            for record in self.get_response_records(
                prompts,
                skip_special_tokens=skip_special_tokens,
            )
        ]

    def get_generation_token_count(self, prompts: list[Prompt]) -> int:
        """
        Number of tokens generated for the given prompts (including any reasoning).

        Used to benchmark raw throughput. Unlike get_responses, this is not reduced to
        the answer text, so it stays meaningful in reasoning mode, where most of the time
        is spent generating the (much longer) reasoning block.
        """
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self._response_max_new_tokens(),
        )

        outputs = cast(LongTensor, outputs)
        return outputs.shape[0] * (
            outputs.shape[1] - cast(Tensor, inputs["input_ids"]).shape[1]
        )

    def _reasoning_close_tag_ids(self) -> list[list[int]]:
        """
        Token IDs of the tag(s) that close a chain-of-thought block, derived from the
        `chain_of_thought_skips` pairs (the part of the closed block that follows the
        initializer).
        """
        if self._cot_close_tag_ids_cache is None:
            tag_ids: list[list[int]] = []
            for initializer, closed_block in self.settings.chain_of_thought_skips:
                closer = closed_block.removeprefix(initializer)
                if not closer:
                    continue
                ids = self.tokenizer.encode(closer, add_special_tokens=False)
                if ids:
                    tag_ids.append(list(ids))
            self._cot_close_tag_ids_cache = tag_ids
        return self._cot_close_tag_ids_cache

    def _reasoning_start_markers(self) -> list[str]:
        """
        The strings that open a chain-of-thought block, i.e. the model's thinking-start
        marker, taken from the initializer part of each `chain_of_thought_skips` pair.
        """
        return [initializer for initializer, _ in self.settings.chain_of_thought_skips]

    def _special_token_ids(self) -> set[int]:
        if self._special_token_ids_cache is None:
            self._special_token_ids_cache = set(self.tokenizer.all_special_ids)
        return self._special_token_ids_cache

    def _eos_token_ids(self) -> list[int]:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return []
        return list(eos) if isinstance(eos, (list, tuple)) else [eos]

    def decode_tokens(self, ids: list[int]) -> str:
        """
        Decode raw token IDs without dropping special tokens, so that markers built from
        special tokens (e.g. channel tags) can still be matched.
        """
        return cast(str, self.tokenizer.decode(ids, skip_special_tokens=False))

    def build_chat_prompts(self, prompts: list[Prompt]) -> list[str]:
        """
        Render prompts through the chat template. Reused by alternative inference
        backends so prompt construction stays identical to the HuggingFace path.
        """
        return self._build_chat_prompts(prompts)

    def response_max_new_tokens(self) -> int:
        """
        Maximum number of tokens a full response generation may produce, including the
        reasoning phase in reasoning mode.
        """
        return self._response_max_new_tokens()

    def answer_and_capture_ids(
        self,
        prompt_ids: list[int],
        generated_ids: list[int],
    ) -> tuple[list[int], list[int], bool]:
        """
        Split generated IDs into answer IDs, the capture position IDs, and an
        "unfinished" flag. Thin wrapper over the internal helper for backends that locate
        the capture position themselves.
        """
        return self._answer_and_capture_ids(prompt_ids, generated_ids)

    def first_content_ids(
        self,
        prompt_ids: list[int],
        generated_ids: list[int],
    ) -> list[int]:
        """
        Token IDs up to the first content-bearing token, skipping the model's
        thinking-start marker. Thin wrapper over the internal helper.
        """
        return self._first_content_ids(prompt_ids, generated_ids)

    def _first_content_ids(
        self, prompt_ids: list[int], generated_ids: list[int]
    ) -> list[int]:
        """
        Token IDs up to (but not including) the first token that is not part of the
        model's thinking-start marker. The distribution captured there is over the first
        token that actually carries content, instead of the (near-deterministic) marker.
        """
        skip = reasoning_start_skip_length(
            prompt_ids,
            generated_ids,
            self._reasoning_start_markers(),
            lambda ids: cast(
                str, self.tokenizer.decode(ids, skip_special_tokens=False)
            ),
            self._special_token_ids(),
        )
        return prompt_ids + generated_ids[:skip]

    def _answer_start_index(self, generated_ids: list[int]) -> int | None:
        """
        Index of the first token after the earliest chain-of-thought closing tag in
        `generated_ids`, or None if no closing tag is present.
        """
        answer_start = None
        for tag_ids in self._reasoning_close_tag_ids():
            tag_length = len(tag_ids)
            for index in range(len(generated_ids) - tag_length + 1):
                if generated_ids[index : index + tag_length] == tag_ids:
                    end = index + tag_length
                    if answer_start is None or end < answer_start:
                        answer_start = end
                    break
        return answer_start

    def _answer_content_start_index(
        self, generated_ids: list[int], answer_start: int
    ) -> int:
        """
        Index of the first token after the reasoning block that carries answer content.

        Chat templates typically insert a whitespace separator between the closing tag
        and the answer (Qwen3, for example, emits "\\n\\n"). Anchoring at the token
        immediately after the tag would capture the near-deterministic separator token
        instead of the first answer token, which makes the answer-anchored KL collapse
        to zero because both the base and modified models are almost certain to emit the
        same newline. Returns `len(generated_ids)` when the answer is empty.
        """
        index = answer_start
        while index < len(generated_ids):
            if self.decode_tokens(generated_ids[index : index + 1]).strip():
                break
            index += 1
        return index

    def _answer_token_ids_with_status(
        self, generated_ids: list[int]
    ) -> tuple[list[int], bool]:
        """
        Return the answer token IDs and an "unfinished" flag. The flag is True when the
        reasoning block never closed (no closing tag found), i.e. the answer is empty
        because generation was cut off mid-reasoning.

        If no closing tags are configured at all, tag detection is impossible, so the
        full generation is returned with unfinished=False.
        """
        if not self._reasoning_close_tag_ids():
            return generated_ids[: self.settings.max_response_length], False

        answer_start = self._answer_start_index(generated_ids)
        if answer_start is None:
            # Reasoning never finished: no answer was produced.
            return [], True

        content_start = self._answer_content_start_index(generated_ids, answer_start)
        return (
            generated_ids[content_start:][: self.settings.max_response_length],
            False,
        )

    def _answer_token_ids(self, generated_ids: list[int]) -> list[int]:
        """
        Given the token IDs generated for a single prompt, return the token IDs of the
        final answer only: everything after the first chain-of-thought closing tag and
        the template's whitespace separator, truncated to max_response_length.

        If the model never closed its reasoning block (no closing tag found, e.g. the
        reasoning budget was exhausted mid-thought), the response is treated as a
        refusal and an empty result is returned. The refusal scorer classifies empty
        responses as refusals, so unfinished reasoning is never mistaken for compliance.
        """
        return self._answer_token_ids_with_status(generated_ids)[0]

    def _answer_and_capture_ids(
        self,
        prompt_ids: list[int],
        generated_ids: list[int],
    ) -> tuple[list[int], list[int], bool]:
        """
        Split a generated sequence into the final answer token IDs and the token IDs up
        to the first answer token (the logit/residual capture position), plus an
        "unfinished" flag. The capture position skips the whitespace separator that chat
        templates place between the closing tag and the answer, so it anchors at the
        first token that carries answer content. When the reasoning block never closed,
        the capture position falls back to the end of the generation.
        """
        answer_ids, unfinished = self._answer_token_ids_with_status(generated_ids)

        if not self._reasoning_close_tag_ids():
            capture_ids = prompt_ids + answer_ids
        else:
            answer_start = self._answer_start_index(generated_ids)
            if answer_start is None:
                capture_ids = prompt_ids + generated_ids
            else:
                content_start = self._answer_content_start_index(
                    generated_ids, answer_start
                )
                capture_ids = prompt_ids + generated_ids[:content_start]

        return answer_ids, capture_ids, unfinished

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [
            record.response
            for record in self.get_response_records_batched(
                prompts,
                skip_special_tokens=skip_special_tokens,
            )
        ]

    def get_response_records_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        capture: CaptureRequest | None = None,
    ) -> list[GenerationRecord]:
        records: list[GenerationRecord] = []
        unfinished = 0
        repetition_aborted = 0
        max_generated = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_records = self.get_response_records(
                batch,
                skip_special_tokens=skip_special_tokens,
                capture=capture,
            )
            records.extend(batch_records)
            unfinished += sum(record.unfinished for record in batch_records)
            repetition_aborted += sum(
                record.repetition_aborted for record in batch_records
            )
            max_generated = max(
                max_generated,
                max(
                    (record.generated_token_count for record in batch_records),
                    default=0,
                ),
            )

        if repetition_aborted:
            print()
            print(
                f"[yellow]Warning:[/] {repetition_aborted}/{len(prompts)} responses were "
                "stopped early because they fell into a repetition loop. Broken models "
                "otherwise generate the full token budget of gibberish."
            )

        if unfinished:
            budget = self._response_max_new_tokens()
            if max_generated >= budget:
                reason = (
                    f"generation reached the {budget}-token budget, so reasoning was cut "
                    "off before the closing tag"
                )
            else:
                reason = (
                    f"generation stopped at {max_generated} tokens, below the "
                    f"{budget}-token budget, so the model ended its turn without the "
                    "expected closing tag"
                )
            print()
            print(
                f"[yellow]Warning:[/] {unfinished}/{len(prompts)} responses have an empty "
                f"answer because the reasoning block never closed: {reason}. Empty "
                "responses are counted as refusals."
            )

        if (
            repetition_aborted
            and self.settings.abort_broken_trials
            and repetition_aborted / len(prompts)
            >= self.settings.abort_broken_trial_fraction
        ):
            raise RepetitionLoopError(
                f"{repetition_aborted}/{len(prompts)} responses fell into repetition "
                "loops, indicating a broken model"
            )

        return records

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
        capture_ids: list[list[int]] | None = None,
    ) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        inputs = (
            self._capture_inputs_from_ids(capture_ids)
            if capture_ids is not None
            else self._capture_inputs(prompts)
        )
        outputs = self._generate_from_inputs(
            inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

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
        residuals = residuals.to(torch.float32)

        if 0 <= winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
        capture_ids: list[list[int]] | None = None,
    ) -> Tensor:
        prompt_batches = batchify(prompts, self.settings.capture_batch_size)
        capture_id_batches: list[list[list[int]] | None] = (
            [None] * len(prompt_batches)
            if capture_ids is None
            else list(batchify(capture_ids, self.settings.capture_batch_size))
        )

        residuals = [
            self.get_residuals(
                batch,
                winsorization_quantile=winsorization_quantile,
                capture_ids=ids,
            )
            for batch, ids in zip(prompt_batches, capture_id_batches)
        ]

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
        capture_ids: list[list[int]] | None = None,
    ) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        prompt_batches = batchify(prompts, self.settings.capture_batch_size)
        capture_id_batches: list[list[list[int]] | None] = (
            [None] * len(prompt_batches)
            if capture_ids is None
            else list(batchify(capture_ids, self.settings.capture_batch_size))
        )

        running_sum = None
        total_count = 0

        for batch, ids in zip(prompt_batches, capture_id_batches):
            batch_residuals = self.get_residuals(
                batch,
                winsorization_quantile=winsorization_quantile,
                capture_ids=ids,
            )

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def get_module_io(
        self,
        prompts: list[Prompt],
        capture_ids: list[list[int]] | None = None,
    ) -> ModuleIO:
        # The list contains one element per layer.
        # Each element maps from the component name to a (possibly sparse) mapping
        # from the module index to an (input, output) tuple containing the I/O
        # tensors of shape (prompt, component).
        module_io: ModuleIO = []

        def get_hook(
            layer_index: int,
            component: str,
            module_index: int,
        ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
            def hook(
                module: Module,
                inputs: tuple[Tensor, ...],
                outputs: Tensor,
            ) -> None:
                if len(module_io) == layer_index:
                    # First invocation of the hook for this layer.
                    module_io.append({})

                # Layers are invoked in order during inference,
                # so this should always hold.
                assert len(module_io) == layer_index + 1

                if component not in module_io[layer_index]:
                    module_io[layer_index][component] = {}

                # Each module should be invoked at most once per inference step.
                assert module_index not in module_io[layer_index][component]

                # inputs[0] and outputs have shape (prompt, position, component),
                # so this extracts the input/output at the end of each prompt.
                # Move to CPU to decouple from device assignments, which can
                # change between model reloads in multi-GPU configurations.
                input = inputs[0][:, -1, :].detach().clone().cpu()
                output = outputs[:, -1, :].detach().clone().cpu()

                # The modules associated with a component (e.g. expert MLPs)
                # are not necessarily invoked in order, nor are all of them
                # necessarily invoked in each inference step, so we cannot
                # use a list here.
                module_io[layer_index][component][module_index] = (input, output)

            return hook

        # Compute the capture inputs before registering the hooks below: in reasoning
        # mode this generates the model's reasoning to locate the answer position, and
        # running that generation with the hooks active would make them fire once per
        # generation step instead of once per capture. If the capture token IDs are
        # already available (e.g. from get_response_records), no generation is needed.
        inputs = (
            self._capture_inputs_from_ids(capture_ids)
            if capture_ids is not None
            else self._capture_inputs(prompts)
        )

        hook_handles: list[RemovableHandle] = []

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    hook_handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )

        self._generate_from_inputs(inputs, max_new_tokens=1, use_cache=False)

        for hook_handle in hook_handles:
            hook_handle.remove()

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
        capture_ids: list[list[int]] | None = None,
    ) -> ModuleIO:
        # Aggregating batch results is more complicated for module I/O
        # than for other get_*_batched methods, because the structure of the results
        # might differ between batches, as whether individual modules activate
        # can depend on the prompt (in particular for MoE models).
        # In practice, inhomogeneous results should be very rare, but to be fully
        # generic, this logic is required.
        prompt_batches = batchify(prompts, self.settings.capture_batch_size)
        capture_id_batches: list[list[list[int]] | None] = (
            [None] * len(prompt_batches)
            if capture_ids is None
            else list(batchify(capture_ids, self.settings.capture_batch_size))
        )
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch, ids)
            for batch, ids in zip(prompt_batches, capture_id_batches)
        ]

        module_io: ModuleIO = []

        for layer_index in range(len(self.get_layers())):
            module_io.append({})

            for module_io_batch in module_io_batches:
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}

                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            # This is a placeholder; the actual aggregation happens below.
                            # We need to iterate over the batches twice because we don't
                            # know in advance which components and module indices are present.
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    input = torch.cat(
                        [input_output[0] for input_output in inputs_outputs],
                        dim=0,
                    )
                    output = torch.cat(
                        [input_output[1] for input_output in inputs_outputs],
                        dim=0,
                    )

                    # The key already exists, and replacing existing values
                    # in a dictionary while iterating over the same dictionary
                    # is safe in Python.
                    module_io[layer_index][component][module_index] = (input, output)

        return module_io

    def _logits_from_inputs(self, inputs: BatchEncoding) -> Tensor:
        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        outputs = self._generate_from_inputs(
            inputs,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu:
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits(
        self,
        prompts: list[Prompt],
        after_reasoning: bool | None = None,
    ) -> Tensor:
        return self._logits_from_inputs(self._capture_inputs(prompts, after_reasoning))

    def get_logits_from_capture_ids(self, capture_ids: list[list[int]]) -> Tensor:
        """
        Logits at the answer-capture position for precomputed capture token IDs (see
        get_response_records). Avoids regenerating the reasoning that produced them.
        """
        return self._logits_from_inputs(self._capture_inputs_from_ids(capture_ids))

    def get_logits_batched(
        self,
        prompts: list[Prompt],
        after_reasoning: bool | None = None,
    ) -> Tensor:
        logits = []

        for batch in batchify(prompts, self.settings.batch_size):
            logits.append(self.get_logits(batch, after_reasoning))

        return torch.cat(logits, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                # Extra template options (e.g. enable_thinking) from the config.
                **self.settings.chat_template_kwargs,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
