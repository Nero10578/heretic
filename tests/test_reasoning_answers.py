# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import cast

from heretic.model import Model


class FakeModel:
    """Minimal stand-in exercising Model._answer_token_ids without a real model."""

    def __init__(self, tag_ids: list[list[int]], max_response_length: int) -> None:
        # Pre-populating the cache makes _reasoning_close_tag_ids() skip tokenization.
        self._cot_close_tag_ids_cache = tag_ids
        self.settings = SimpleNamespace(max_response_length=max_response_length)

    def _reasoning_close_tag_ids(self) -> list[list[int]]:
        return self._cot_close_tag_ids_cache

    def _answer_start_index(self, generated_ids: list[int]) -> int | None:
        return Model._answer_start_index(cast(Model, self), generated_ids)

    def _answer_token_ids_with_status(
        self, generated_ids: list[int]
    ) -> tuple[list[int], bool]:
        return Model._answer_token_ids_with_status(cast(Model, self), generated_ids)


class ReasoningAnswerTests(unittest.TestCase):
    def answer(
        self,
        generated: list[int],
        tag_ids: list[list[int]],
        max_response_length: int = 100,
    ) -> list[int]:
        model = cast(Model, FakeModel(tag_ids, max_response_length))
        return Model._answer_token_ids(model, generated)

    def test_extracts_text_after_closing_tag(self) -> None:
        self.assertEqual(self.answer([10, 11, 99, 20, 21], [[99]]), [20, 21])

    def test_truncates_answer_to_max_response_length(self) -> None:
        self.assertEqual(
            self.answer([99, 20, 21, 22, 23], [[99]], max_response_length=2),
            [20, 21],
        )

    def test_unfinished_reasoning_is_treated_as_refusal(self) -> None:
        # No closing tag => empty response => counted as a refusal by the scorer.
        self.assertEqual(self.answer([10, 11, 12], [[99]]), [])

    def test_uses_earliest_closing_tag(self) -> None:
        self.assertEqual(self.answer([98, 1, 99, 2], [[99], [98]]), [1, 99, 2])

    def test_no_tags_configured_returns_generation(self) -> None:
        self.assertEqual(self.answer([1, 2, 3], [], max_response_length=2), [1, 2])

    def test_unfinished_reasoning_reports_status(self) -> None:
        model = cast(Model, FakeModel([[99]], 100))
        ids, unfinished = Model._answer_token_ids_with_status(model, [10, 11, 12])
        self.assertEqual(ids, [])
        self.assertTrue(unfinished)

    def test_finished_reasoning_reports_status(self) -> None:
        model = cast(Model, FakeModel([[99]], 100))
        ids, unfinished = Model._answer_token_ids_with_status(model, [99, 20, 21])
        self.assertEqual(ids, [20, 21])
        self.assertFalse(unfinished)

    def test_no_tags_reports_not_unfinished(self) -> None:
        model = cast(Model, FakeModel([], 100))
        ids, unfinished = Model._answer_token_ids_with_status(model, [1, 2, 3])
        self.assertEqual(ids, [1, 2, 3])
        self.assertFalse(unfinished)

    def test_answer_start_index_finds_position_after_tag(self) -> None:
        model = cast(Model, FakeModel([[99]], 100))
        self.assertEqual(Model._answer_start_index(model, [10, 99, 20]), 2)

    def test_answer_start_index_is_none_without_tag(self) -> None:
        model = cast(Model, FakeModel([[99]], 100))
        self.assertIsNone(Model._answer_start_index(model, [10, 20, 30]))


if __name__ == "__main__":
    unittest.main()
