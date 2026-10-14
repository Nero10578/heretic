# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic.model import reasoning_start_skip_length

# A tiny fake tokenizer: maps token IDs to text and decodes by concatenation.
TOKENS = {1: "<a>", 2: "thought", 3: "\n", 4: "hello", 5: "<b>"}
MARKER = "<a>thought\n"


def decode(ids: list[int]) -> str:
    return "".join(TOKENS[i] for i in ids)


class ReasoningStartSkipTests(unittest.TestCase):
    def test_skips_marker_generated_by_model(self) -> None:
        self.assertEqual(
            reasoning_start_skip_length([5], [1, 2, 3, 4], [MARKER], decode, set()),
            3,
        )

    def test_skips_marker_opened_by_template(self) -> None:
        # Prompt already emitted "<a>"; the model generates the rest of the marker.
        self.assertEqual(
            reasoning_start_skip_length([5, 1], [2, 3, 4], [MARKER], decode, set()),
            2,
        )

    def test_marker_fully_in_template_means_content_start(self) -> None:
        # Template emitted the whole marker, so the first generated token is content.
        self.assertEqual(
            reasoning_start_skip_length([5, 1, 2, 3], [4, 4], [MARKER], decode, set()),
            0,
        )

    def test_no_marker_skips_leading_special_tokens(self) -> None:
        self.assertEqual(
            reasoning_start_skip_length([5], [1, 1, 4], [], decode, {1}),
            2,
        )

    def test_content_start_returns_zero(self) -> None:
        self.assertEqual(
            reasoning_start_skip_length([5], [4, 4], [MARKER], decode, {1}),
            0,
        )


if __name__ == "__main__":
    unittest.main()
