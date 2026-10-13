# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import cast

from heretic.config import Settings as HereticSettings
from heretic.model import GenerationRecord, Model
from heretic.plugin import Context
from heretic.utils import Prompt


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def get_response_records_batched(
        self, prompts: list[Prompt], skip_special_tokens: bool = False
    ) -> list[GenerationRecord]:
        return [
            GenerationRecord(
                response=response,
                capture_ids=[],
                unfinished=False,
                generated_token_count=0,
            )
            for response in self.responses
        ]


class PromptFilterTests(unittest.TestCase):
    def make_ctx(self, responses: list[str]) -> Context:
        settings = cast(HereticSettings, SimpleNamespace())
        return Context(settings, cast(Model, FakeModel(responses)))

    def test_keep_refused_keeps_only_refusals(self) -> None:
        ctx = self.make_ctx(["I'm sorry, I cannot help.", "Sure, here you go."])
        prompts = [Prompt("s", "a"), Prompt("s", "b")]
        kept = ctx.filter_prompts_by_refusal(
            prompts, keep_refused=True, refusal_markers=["sorry", "i cannot"]
        )
        self.assertEqual([p.user for p in kept], ["a"])

    def test_drop_refused_drops_refusals(self) -> None:
        ctx = self.make_ctx(["I'm sorry, I cannot help.", "Sure, here you go."])
        prompts = [Prompt("s", "a"), Prompt("s", "b")]
        kept = ctx.filter_prompts_by_refusal(
            prompts, keep_refused=False, refusal_markers=["sorry", "i cannot"]
        )
        self.assertEqual([p.user for p in kept], ["b"])

    def test_empty_response_counts_as_refusal(self) -> None:
        ctx = self.make_ctx([""])
        prompts = [Prompt("s", "a")]
        kept = ctx.filter_prompts_by_refusal(
            prompts, keep_refused=False, refusal_markers=["sorry"]
        )
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
