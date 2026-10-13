# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from unittest import mock

from heretic.model import Model


class AttentionCandidateTests(unittest.TestCase):
    def test_auto_without_flash_attn_uses_library_default(self) -> None:
        with (
            mock.patch("heretic.model.is_flash_attn_2_available", return_value=False),
            mock.patch("heretic.model.is_flash_attn_3_available", return_value=False),
        ):
            self.assertEqual(Model._attention_candidates("auto"), [None])

    def test_auto_with_flash_attn_prefers_flash(self) -> None:
        with mock.patch("heretic.model.is_flash_attn_2_available", return_value=True):
            self.assertEqual(
                Model._attention_candidates("auto"),
                ["flash_attention_2", "sdpa"],
            )

    def test_explicit_sdpa_is_used_as_is(self) -> None:
        self.assertEqual(Model._attention_candidates("sdpa"), ["sdpa"])

    def test_explicit_flash_falls_back_to_sdpa_then_default(self) -> None:
        self.assertEqual(
            Model._attention_candidates("flash_attention_4"),
            ["flash_attention_4", "sdpa", None],
        )


if __name__ == "__main__":
    unittest.main()
