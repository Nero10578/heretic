# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import Any, cast

import torch

from heretic.model import (
    CaptureRequest,
    Model,
    _GenerationCapture,
    merge_module_io_rows,
    split_module_io_rows,
)


class TinyModelShim:
    """Minimal stand-in exposing what _GenerationCapture reads from the model."""

    def __init__(
        self,
        close_tag_ids: list[list[int]],
        markers: list[str],
        layer_count: int = 2,
        token_text: dict[int, str] | None = None,
        modules: dict[str, list[Any]] | None = None,
    ) -> None:
        self._close_tag_ids = close_tag_ids
        self._markers = markers
        self._layer_count = layer_count
        self._token_text = token_text or {}
        self._modules = modules or {}

    def _reasoning_close_tag_ids(self) -> list[list[int]]:
        return self._close_tag_ids

    def _reasoning_start_markers(self) -> list[str]:
        return self._markers

    def get_layers(self) -> list[int]:
        return list(range(self._layer_count))

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Any]]:
        return self._modules if layer_index == 0 else {}

    def output_device(self) -> torch.device:
        return torch.device("cpu")

    def decode_tokens(self, ids: list[int]) -> str:
        return "".join(self._token_text.get(i, f"<{i}>") for i in ids)


def make_capture(
    shim: TinyModelShim,
    capture_logits: bool = True,
    capture_module_io: bool = False,
    prompt_ids: list[list[int]] | None = None,
) -> _GenerationCapture:
    return _GenerationCapture(
        model=cast(Model, shim),
        prompt_length=1,
        prompt_ids=prompt_ids if prompt_ids is not None else [[7]],
        capture_logits=capture_logits,
        capture_module_io=capture_module_io,
    )


def scores_row(value: float, width: int = 3) -> torch.Tensor:
    return torch.full((1, width), value)


def expected_row(value: float, width: int = 3) -> torch.Tensor:
    return torch.full((width,), value)


class CapturePositionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Token 5 decodes to the start marker, 9 and 10 are the closing tag.
        self.shim = TinyModelShim(
            close_tag_ids=[[9, 10]],
            markers=["<t>"],
            token_text={5: "<t>"},
        )

    def test_content_and_answer_positions(self) -> None:
        capture = make_capture(self.shim)

        capture.observe(torch.tensor([[7, 5]]), scores_row(1.0))
        self.assertEqual(capture.content_positions, {0: 1})
        self.assertEqual(capture.answer_positions, {})
        self.assertTrue(torch.equal(capture.content_logits[0], expected_row(1.0)))

        capture.observe(torch.tensor([[7, 5, 9]]), scores_row(2.0))
        self.assertEqual(capture.answer_positions, {})

        # The closing tag completes here; the capture waits for the first
        # content-bearing token so it can anchor past a whitespace separator.
        capture.observe(torch.tensor([[7, 5, 9, 10]]), scores_row(3.0))
        self.assertEqual(capture.answer_positions, {})

        capture.observe(torch.tensor([[7, 5, 9, 10, 11]]), scores_row(4.0))
        self.assertEqual(capture.answer_positions, {0: 3})
        self.assertTrue(torch.equal(capture.answer_logits[0], expected_row(3.0)))

    def test_positions_are_not_overwritten(self) -> None:
        capture = make_capture(self.shim)
        capture.observe(torch.tensor([[7, 5]]), scores_row(1.0))
        # A later step must not change the content capture (position stays 1, value 1.0).
        capture.observe(torch.tensor([[7, 5, 9, 10]]), scores_row(8.0))
        capture.observe(torch.tensor([[7, 5, 9, 10, 11]]), scores_row(9.0))
        self.assertEqual(capture.content_positions, {0: 1})
        self.assertTrue(torch.equal(capture.content_logits[0], expected_row(1.0)))
        self.assertEqual(capture.answer_positions, {0: 3})
        self.assertTrue(torch.equal(capture.answer_logits[0], expected_row(8.0)))

    def test_skips_whitespace_separator_before_answer(self) -> None:
        # Qwen3-style separator: the tokens after the closing tag are newlines.
        shim = TinyModelShim(
            close_tag_ids=[[9]],
            markers=[],
            token_text={198: "\n", 4: "hello"},
        )
        capture = make_capture(shim)

        capture.observe(torch.tensor([[7, 5, 9]]), scores_row(1.0))
        capture.observe(torch.tensor([[7, 5, 9, 198]]), scores_row(2.0))
        capture.observe(torch.tensor([[7, 5, 9, 198, 198]]), scores_row(3.0))
        self.assertEqual(capture.answer_positions, {})

        capture.observe(torch.tensor([[7, 5, 9, 198, 198, 4]]), scores_row(4.0))
        self.assertEqual(capture.answer_positions, {0: 4})
        # The captured distribution is the one that predicted "hello".
        self.assertTrue(torch.equal(capture.answer_logits[0], expected_row(3.0)))

    def test_answer_module_io_uses_previous_step(self) -> None:
        shim = TinyModelShim(close_tag_ids=[[9]], markers=[], layer_count=1)
        capture = _GenerationCapture(
            model=cast(Model, shim),
            prompt_length=1,
            prompt_ids=[[7]],
            capture_logits=True,
            capture_module_io=True,
        )
        capture._expected_io_count = 1

        def make_io(value: float) -> dict[int, dict[str, dict[int, Any]]]:
            return {
                0: {
                    "proj": {
                        0: (
                            torch.full((1, 2), value),
                            torch.full((1, 2), value),
                        )
                    }
                }
            }

        # The closing tag completes here, so this step's I/O ends at the tag token.
        capture._io = make_io(1.0)
        capture.observe(torch.tensor([[7, 5, 9]]), scores_row(1.0))

        # A content token appears; the answer must use the previous step's I/O.
        capture._io = make_io(9.0)
        capture.observe(torch.tensor([[7, 5, 9, 4]]), scores_row(2.0))

        self.assertEqual(capture.answer_positions, {0: 2})
        input, output = capture.module_io_rows[0][0]["proj"][0]
        self.assertTrue(torch.equal(input, torch.full((1, 2), 1.0)))
        self.assertTrue(torch.equal(output, torch.full((1, 2), 1.0)))

    def test_unfinished_row_is_not_captured(self) -> None:
        capture = make_capture(self.shim)
        for step, row in enumerate([[7, 5], [7, 5, 9], [7, 5, 9, 11]], start=1):
            capture.observe(torch.tensor([row]), scores_row(float(step)))
        self.assertEqual(capture.answer_positions, {})
        self.assertNotIn(0, capture.answer_logits)

    def test_content_probe_window_limits_marker_search(self) -> None:
        # The marker only appears after the probe window, so it must not be captured.
        shim = TinyModelShim(
            close_tag_ids=[[9]], markers=["<t>"], token_text={5: "<t>"}
        )
        capture = make_capture(shim)
        row = [7] + [1] * 8 + [5]
        capture.observe(torch.tensor([row]), scores_row(4.0))
        self.assertEqual(capture.content_positions, {})

    def test_batch_rows_tracked_independently(self) -> None:
        capture = make_capture(self.shim, prompt_ids=[[7], [7]])
        # Row 0 closes at step 3, row 1 never closes.
        capture.observe(torch.tensor([[7, 5, 9], [7, 5, 1]]), torch.full((2, 3), 1.0))
        capture.observe(
            torch.tensor([[7, 5, 9, 10], [7, 5, 1, 1]]), torch.full((2, 3), 2.0)
        )
        capture.observe(
            torch.tensor([[7, 5, 9, 10, 11], [7, 5, 1, 1, 1]]),
            torch.full((2, 3), 3.0),
        )
        self.assertEqual(capture.answer_positions, {0: 3})
        self.assertTrue(torch.equal(capture.answer_logits[0], expected_row(2.0)))
        self.assertNotIn(1, capture.answer_logits)

    def test_module_io_marked_incomplete_is_not_captured(self) -> None:
        # Expected modules exist but none fire, so module I/O must be left uncaptured
        # (the caller then falls back to the separate forward).
        shim = TinyModelShim(
            close_tag_ids=[[9]],
            markers=[],
            layer_count=1,
            modules={"proj": [torch.nn.Linear(2, 2)]},
        )
        capture = _GenerationCapture(
            model=cast(Model, shim),
            prompt_length=1,
            prompt_ids=[[7]],
            capture_logits=True,
            capture_module_io=True,
        )
        capture._expected_io_count = 1
        capture.observe(torch.tensor([[7, 5, 9]]), scores_row(1.0))
        capture.observe(torch.tensor([[7, 5, 9, 13]]), scores_row(2.0))
        self.assertEqual(capture.answer_positions, {0: 2})
        self.assertNotIn(0, capture.module_io_rows)


class ModuleIoMergeTests(unittest.TestCase):
    def test_split_then_merge_round_trip(self) -> None:
        module_io = [
            {"a": {0: (torch.rand(3, 4), torch.rand(3, 4))}},
            {"b": {0: (torch.rand(3, 4), torch.rand(3, 4))}},
        ]
        rows = split_module_io_rows(module_io, 3)
        self.assertEqual(len(rows), 3)

        merged = merge_module_io_rows(rows, len(module_io))
        for layer_index, components in enumerate(module_io):
            for component, modules in components.items():
                for module_index, (input, output) in modules.items():
                    self.assertTrue(
                        torch.equal(
                            merged[layer_index][component][module_index][0], input
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            merged[layer_index][component][module_index][1], output
                        )
                    )


class GenerationCaptureEquivalenceTests(unittest.TestCase):
    """
    Verify the invariant the capture relies on: the logits and module I/O seen by a
    logits processor / forward hook at generation step `a` equal a direct forward over
    the first `prompt_length + a` tokens. This is what lets capturing during generation
    replace the separate capture forward.
    """

    # Typed as Any so the test is not coupled to transformers' generate() protocol typing.
    model: Any

    def setUp(self) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            eos_token_id=None,
        )
        self.model = LlamaForCausalLM(config)
        self.model.eval()

    def _direct_forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        io: dict[str, Any] = {}

        def hook(name: str):
            def inner(module: Any, inputs: Any, outputs: Any) -> None:
                io[name] = (
                    inputs[0][:, -1, :].detach().clone(),
                    outputs[:, -1, :].detach().clone(),
                )

            return inner

        handles = []
        for layer_index, layer in enumerate(self.model.model.layers):
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    handles.append(
                        module.register_forward_hook(hook(f"{layer_index}.{name}"))
                    )

        with torch.no_grad():
            output = self.model(input_ids=input_ids, use_cache=False)

        for handle in handles:
            handle.remove()

        return output.logits[:, -1, :], io

    def test_scores_and_module_io_match_direct_forward(self) -> None:
        from transformers.generation.logits_process import (
            LogitsProcessor,
            LogitsProcessorList,
        )

        prompt = torch.tensor([[5, 6, 7, 8, 9]])
        samples: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = []
        io: dict[str, Any] = {}

        class Recorder(LogitsProcessor):
            def __call__(
                self, input_ids: torch.Tensor, scores: torch.Tensor
            ) -> torch.Tensor:
                samples.append(
                    (
                        input_ids.detach().clone(),
                        scores.detach().clone(),
                        {key: value for key, value in io.items()},
                    )
                )
                return scores

        def hook(name: str):
            def inner(module: Any, inputs: Any, outputs: Any) -> None:
                io[name] = (
                    inputs[0][:, -1, :].detach().clone(),
                    outputs[:, -1, :].detach().clone(),
                )

            return inner

        handles = []
        for layer_index, layer in enumerate(self.model.model.layers):
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    handles.append(
                        module.register_forward_hook(hook(f"{layer_index}.{name}"))
                    )

        with torch.no_grad():
            self.model.generate(
                prompt,
                max_new_tokens=5,
                do_sample=False,
                logits_processor=LogitsProcessorList([Recorder()]),
            )

        for handle in handles:
            handle.remove()

        self.assertGreater(len(samples), 0)
        for input_ids, scores, captured_io in samples:
            reference_logits, reference_io = self._direct_forward(input_ids)
            self.assertLess(
                (reference_logits - scores).abs().max().item(),
                1e-4,
            )
            self.assertEqual(set(captured_io.keys()), set(reference_io.keys()))
            for key in reference_io:
                for index in range(2):
                    self.assertLess(
                        (captured_io[key][index] - reference_io[key][index])
                        .abs()
                        .max()
                        .item(),
                        1e-4,
                    )


class CaptureRequestTests(unittest.TestCase):
    def test_defaults_are_disabled(self) -> None:
        request = CaptureRequest()
        self.assertFalse(request.logits)
        self.assertFalse(request.module_io)


if __name__ == "__main__":
    unittest.main()
