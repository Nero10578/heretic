# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

import torch

from heretic.model import RepetitionStoppingLogitsProcessor


class RepetitionProcessorTests(unittest.TestCase):
    def make(
        self,
        prompt_length: int = 1,
        period_max: int = 4,
        min_repeats: int = 3,
    ) -> RepetitionStoppingLogitsProcessor:
        return RepetitionStoppingLogitsProcessor(
            prompt_length=prompt_length,
            eos_token_ids=[9],
            pad_token_id=0,
            period_max=period_max,
            min_repeats=min_repeats,
        )

    def test_forces_eos_on_repeated_single_token(self) -> None:
        processor = self.make()
        scores = torch.zeros((1, 10))
        processor(torch.tensor([[7, 5, 5, 5, 5]]), scores)
        self.assertEqual(processor.aborted_rows, {0})
        self.assertTrue(torch.isinf(scores[0, 9]))

    def test_forces_eos_on_period_two_cycle(self) -> None:
        processor = self.make()
        processor(torch.tensor([[7, 5, 6, 5, 6, 5, 6]]), torch.zeros((1, 10)))
        self.assertEqual(processor.aborted_rows, {0})

    def test_ignores_finished_padded_rows(self) -> None:
        # After EOS the row is padded, so its tail is identical pad tokens. This must
        # not be mistaken for a repetition loop.
        processor = self.make()
        processor(torch.tensor([[7, 5, 5, 5, 5, 9, 0, 0, 0]]), torch.zeros((1, 10)))
        self.assertEqual(processor.aborted_rows, set())

    def test_does_not_abort_normal_output(self) -> None:
        processor = self.make()
        scores = torch.zeros((1, 10))
        processor(torch.tensor([[7, 1, 2, 3, 4, 5, 6]]), scores)
        self.assertEqual(processor.aborted_rows, set())
        self.assertFalse(torch.isinf(scores).any())

    def test_ignores_repeated_prompt_tokens(self) -> None:
        # The repetition is in the prompt, not in the generation.
        processor = self.make(prompt_length=4)
        processor(torch.tensor([[5, 5, 5, 5, 1, 2, 3]]), torch.zeros((1, 10)))
        self.assertEqual(processor.aborted_rows, set())

    def test_ignores_generation_shorter_than_needed(self) -> None:
        processor = self.make()
        processor(torch.tensor([[7, 5, 5]]), torch.zeros((1, 10)))
        self.assertEqual(processor.aborted_rows, set())

    def test_only_repetitive_rows_are_aborted(self) -> None:
        processor = self.make()
        scores = torch.zeros((2, 10))
        # Row 0 loops; row 1 is normal.
        processor(
            torch.tensor([[7, 5, 5, 5, 5], [7, 1, 2, 3, 4]]),
            scores,
        )
        self.assertEqual(processor.aborted_rows, {0})
        self.assertTrue(torch.isinf(scores[0, 9]))
        self.assertFalse(torch.isinf(scores[1]).any())


if __name__ == "__main__":
    unittest.main()
