# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import cast

import torch

from heretic.config import Settings as HereticSettings
from heretic.model import GenerationRecord, Model
from heretic.plugin import Context
from heretic.utils import Prompt


class FakeModel:
    """Records which prompts it is asked to generate, so reuse can be asserted."""

    def __init__(self) -> None:
        self.generated_batches: list[list[str]] = []

    def get_response_records_batched(
        self, prompts: list[Prompt], skip_special_tokens: bool = False
    ) -> list[GenerationRecord]:
        self.generated_batches.append([prompt.user for prompt in prompts])
        return [
            GenerationRecord(
                response="response",
                capture_ids=[1, 2, 3],
                unfinished=False,
                generated_token_count=1,
            )
            for _ in prompts
        ]

    def get_logits_from_capture_ids(self, capture_ids: list[list[int]]) -> torch.Tensor:
        assert all(ids == [1, 2, 3] for ids in capture_ids)
        return torch.zeros((len(capture_ids), 4))

    def get_logits_batched(
        self, prompts: list[Prompt], after_reasoning: bool | None = None
    ) -> torch.Tensor:
        return torch.zeros((len(prompts), 4))


class GenerationReuseTests(unittest.TestCase):
    def make_ctx(self) -> tuple[Context, FakeModel]:
        settings = cast(
            HereticSettings,
            SimpleNamespace(batch_size=32, generate_chain_of_thought=True),
        )
        model = FakeModel()
        return Context(settings, cast(Model, model)), model

    def test_responses_and_logits_share_one_generation(self) -> None:
        ctx, model = self.make_ctx()
        prompts = [Prompt("s", "a"), Prompt("s", "b")]

        ctx.get_responses(prompts)
        logits = ctx.get_logits(prompts)

        self.assertEqual(model.generated_batches, [["a", "b"]])
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_partial_overlap_only_generates_missing_prompts(self) -> None:
        ctx, model = self.make_ctx()

        ctx.get_responses([Prompt("s", "a"), Prompt("s", "b")])
        ctx.get_responses([Prompt("s", "b"), Prompt("s", "c")])

        self.assertEqual(model.generated_batches, [["a", "b"], ["c"]])


if __name__ == "__main__":
    unittest.main()
