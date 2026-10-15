# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

from heretic.config import Settings as HereticSettings
from heretic.config import VLLMSettings
from heretic.model import CaptureRequest, GenerationRecord, Model
from heretic.plugin import Context
from heretic.scorer import Scorer
from heretic.scorers.keyword_rate import KeywordRate
from heretic.scorers.kl_divergence import KLDivergence
from heretic.utils import Prompt
from heretic.vllm_backend import (
    VLLMBackend,
    normalize_adapter_target_modules,
    target_module_suffixes,
)


def make_record(
    response: str = "response",
    capture_ids: list[int] | None = None,
    first_content_ids: list[int] | None = None,
    answer_logits: torch.Tensor | None = None,
) -> GenerationRecord:
    return GenerationRecord(
        response=response,
        capture_ids=capture_ids or [],
        first_content_ids=first_content_ids or [],
        unfinished=False,
        repetition_aborted=False,
        generated_token_count=0,
        answer_logits=answer_logits,
    )


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool, Any]] = []

    def get_response_records_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
        capture: Any = None,
    ) -> list[GenerationRecord]:
        self.calls.append(
            ([prompt.user for prompt in prompts], skip_special_tokens, capture)
        )
        return [make_record() for _ in prompts]


class FakeBackend:
    def __init__(self) -> None:
        self.prompts: list[list[str]] = []
        self.captures: list[CaptureRequest] = []
        self.logprob_prompt_calls: list[list[str]] = []
        self.logprob_capture_calls: list[list[list[int]]] = []
        self.answer_logits: torch.Tensor | None = None
        self.synced = 0

    def sync_adapter(self, model: Model) -> None:
        self.synced += 1

    def get_generation_records(
        self, prompts: list[Prompt], *, capture: CaptureRequest
    ) -> list[GenerationRecord]:
        self.prompts.append([prompt.user for prompt in prompts])
        self.captures.append(capture)
        return [
            make_record(
                response="backend",
                capture_ids=[1, 2],
                first_content_ids=[3, 4],
                answer_logits=self.answer_logits,
            )
            for _ in prompts
        ]

    def get_logprobs_prompt(self, prompts: list[Prompt]) -> torch.Tensor:
        self.logprob_prompt_calls.append([prompt.user for prompt in prompts])
        return torch.zeros((len(prompts), 3))

    def get_logprobs_from_capture_ids(
        self, capture_ids: list[list[int]]
    ) -> torch.Tensor:
        self.logprob_capture_calls.append(capture_ids)
        return torch.zeros((len(capture_ids), 3))


class BackendSeamTests(unittest.TestCase):
    def make_settings(self) -> HereticSettings:
        return cast(
            HereticSettings,
            SimpleNamespace(generate_chain_of_thought=False),
        )

    def test_default_backend_generates_with_model(self) -> None:
        model = FakeModel()
        ctx = Context(self.make_settings(), cast(Model, model))

        responses = ctx.get_responses([Prompt("s", "a")])

        self.assertEqual(responses, ["response"])
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(model.calls[0][1])

    def test_injected_backend_replaces_model(self) -> None:
        model = FakeModel()
        backend = FakeBackend()
        ctx = Context(self.make_settings(), cast(Model, model), backend=backend)

        responses = ctx.get_responses([Prompt("s", "a")])

        self.assertEqual(responses, ["backend"])
        self.assertEqual(model.calls, [])
        self.assertEqual(backend.prompts, [["a"]])

    def test_capture_defaults_reach_backend(self) -> None:
        backend = FakeBackend()
        ctx = Context(self.make_settings(), cast(Model, FakeModel()), backend=backend)
        ctx.capture_logits_default = True

        ctx.get_responses([Prompt("s", "a")])

        capture = backend.captures[0]
        self.assertTrue(capture.logits)
        self.assertFalse(capture.module_io)


class LogprobTests(unittest.TestCase):
    def make_settings(self, generate_chain_of_thought: bool) -> HereticSettings:
        return cast(
            HereticSettings,
            SimpleNamespace(
                generate_chain_of_thought=generate_chain_of_thought,
                capture_batch_size=8,
            ),
        )

    def test_prompt_capture_uses_backend(self) -> None:
        backend = FakeBackend()
        ctx = Context(
            self.make_settings(False), cast(Model, FakeModel()), backend=backend
        )

        result = ctx.get_logprobs([Prompt("s", "a")])

        self.assertEqual(tuple(result.shape), (1, 3))
        self.assertEqual(backend.logprob_prompt_calls, [["a"]])

    def test_generation_capture_is_converted_without_fallback(self) -> None:
        backend = FakeBackend()
        backend.answer_logits = torch.zeros(3)
        ctx = Context(
            self.make_settings(True), cast(Model, FakeModel()), backend=backend
        )

        result = ctx.get_logprobs([Prompt("s", "a")], after_reasoning=True)

        self.assertEqual(tuple(result.shape), (1, 3))
        self.assertEqual(backend.logprob_capture_calls, [])
        self.assertAlmostEqual(float(result.logsumexp(dim=-1)), 0.0, places=5)

    def test_missing_capture_falls_back_to_backend(self) -> None:
        backend = FakeBackend()
        ctx = Context(
            self.make_settings(True), cast(Model, FakeModel()), backend=backend
        )

        result = ctx.get_logprobs([Prompt("s", "a")], after_reasoning=True)

        self.assertEqual(tuple(result.shape), (1, 3))
        self.assertEqual(backend.logprob_capture_calls, [[[1, 2]]])


class TargetModuleTests(unittest.TestCase):
    def test_suffixes_are_deduped_and_sorted(self) -> None:
        names = [
            "model.layers.0.self_attn.o_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.0.mlp.down_proj",
        ]
        self.assertEqual(target_module_suffixes(names), ["down_proj", "o_proj"])

    def test_normalize_rewrites_fully_qualified_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_config.json"
            path.write_text(
                json.dumps({"target_modules": ["a.b.o_proj", "a.c.down_proj"]}),
                encoding="utf-8",
            )

            normalize_adapter_target_modules(path)

            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["target_modules"], ["down_proj", "o_proj"])

    def test_normalize_leaves_missing_targets_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_config.json"
            original = {"r": 8, "lora_alpha": 8}
            path.write_text(json.dumps(original), encoding="utf-8")

            normalize_adapter_target_modules(path)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)


class AdapterSyncTests(unittest.TestCase):
    def make_backend(self, model: Any) -> VLLMBackend:
        settings = cast(
            HereticSettings,
            SimpleNamespace(model="m", vllm=VLLMSettings()),
        )
        return VLLMBackend(settings, cast(Model, model))

    def test_no_adapter_when_model_is_not_peft(self) -> None:
        model = SimpleNamespace(model=object())
        backend = self.make_backend(model)
        try:
            backend.sync_adapter(cast(Model, model))
            self.assertIsNone(backend._adapter_request)
        finally:
            backend.close()


class VLLMSettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = VLLMSettings()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.mode, "inprocess")
        self.assertEqual(settings.max_lora_rank, 64)
        self.assertEqual(settings.max_logprobs, -1)
        self.assertEqual(settings.logprobs_mode, "raw_logprobs")

    def test_settings_exposes_vllm_table(self) -> None:
        self.assertIn("vllm", HereticSettings.model_fields)


class InternalsTests(unittest.TestCase):
    def test_default_requires_internals(self) -> None:
        self.assertTrue(Scorer.requires_model_internals.fget(cast(Scorer, object())))

    def test_keyword_rate_does_not_require_internals(self) -> None:
        self.assertFalse(
            KeywordRate.requires_model_internals.fget(cast(KeywordRate, object()))
        )

    def test_kl_divergence_does_not_require_internals(self) -> None:
        self.assertFalse(
            KLDivergence.requires_model_internals.fget(cast(KLDivergence, object()))
        )


if __name__ == "__main__":
    unittest.main()
