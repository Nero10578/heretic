# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch

from heretic.cache import InitCache, environment_key, plugin_namespace, source_hash
from heretic.config import Settings as HereticSettings
from heretic.modifiers.ara import ARA
from heretic.modifiers.ara import Settings as ARASettings
from heretic.plugin import Context
from heretic.scorers.keyword_rate import KeywordRate, Settings
from heretic.scorers.kl_divergence import KLDivergence
from heretic.scorers.kl_divergence import Settings as KLSettings
from heretic.utils import Prompt


class InitCacheTests(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = InitCache(directory, "env", "namespace")
            payload = {
                "prompts": [["system", "user"]],
                "count": 3,
                "tensor": torch.arange(6).reshape(2, 3),
                "nested": {0: {1: (torch.ones(1, 2), torch.zeros(1, 2))}},
            }
            cache.save(payload)

            loaded = cache.load()
            assert loaded is not None
            self.assertEqual(loaded["prompts"], [["system", "user"]])
            self.assertEqual(loaded["count"], 3)
            self.assertTrue(torch.equal(loaded["tensor"], payload["tensor"]))
            self.assertTrue(
                torch.equal(loaded["nested"][0][1][0], payload["nested"][0][1][0])
            )

    def test_miss_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(InitCache(directory, "env", "missing").load())

    def test_corrupt_entry_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = InitCache(directory, "env", "namespace")
            cache.path.parent.mkdir(parents=True, exist_ok=True)
            cache.path.write_bytes(b"not a torch file")
            self.assertIsNone(cache.load())


class CacheKeyTests(unittest.TestCase):
    def test_environment_key_tracks_relevant_settings(self) -> None:
        base = HereticSettings.model_construct(
            model="m",
            generate_chain_of_thought=True,
            max_thought_length=4096,
            n_trials=100,
        )

        # A setting that does not affect initialization must not invalidate the cache.
        same_relevant = HereticSettings.model_construct(
            model="m",
            generate_chain_of_thought=True,
            max_thought_length=4096,
            n_trials=999,
        )
        self.assertEqual(environment_key(base), environment_key(same_relevant))

        # A setting that does affect initialization must invalidate the cache.
        changed = HereticSettings.model_construct(
            model="m",
            generate_chain_of_thought=True,
            max_thought_length=8192,
            n_trials=100,
        )
        self.assertNotEqual(environment_key(base), environment_key(changed))

    def test_plugin_namespace_depends_on_settings(self) -> None:
        first = plugin_namespace("KeywordRate", Settings(print_responses=False))
        self.assertEqual(
            first, plugin_namespace("KeywordRate", Settings(print_responses=False))
        )
        self.assertNotEqual(
            first, plugin_namespace("KeywordRate", Settings(print_responses=True))
        )
        self.assertNotEqual(
            first, plugin_namespace("Other", Settings(print_responses=False))
        )
        self.assertNotIn("/", first)
        self.assertNotIn(":", first)

    def test_source_hash_is_stable(self) -> None:
        self.assertEqual(source_hash(), source_hash())


class PluginCacheTests(unittest.TestCase):
    def test_cache_disabled(self) -> None:
        scorer = KeywordRate(
            heretic_settings=HereticSettings.model_construct(init_cache=False),
            settings=Settings(),
        )
        self.assertIsNone(scorer.init_cache())

    def test_cache_path_uses_directory_and_environment(self) -> None:
        settings = HereticSettings.model_construct(
            model="m",
            init_cache=True,
            init_cache_dir="some/dir",
        )
        scorer = KeywordRate(
            heretic_settings=settings,
            settings=Settings(),
        )
        cache = scorer.init_cache()
        assert cache is not None
        self.assertEqual(cache.directory, Path("some/dir"))
        self.assertEqual(cache.path.parent.parent, Path("some/dir"))
        self.assertEqual(cache.path.parent.name, environment_key(settings))


class _FakeModelShim:
    def __init__(self) -> None:
        self.applied_lora: list[int] = []

    def output_device(self) -> torch.device:
        return torch.device("cpu")

    def apply_lora(self, rank: int) -> None:
        self.applied_lora.append(rank)


class _FakeContext:
    def __init__(self, model: _FakeModelShim) -> None:
        self._model = model
        self.loaded: list[object] = []

    def get_model(self) -> _FakeModelShim:
        return self._model

    def load_prompts(self, specification: object) -> list[Prompt]:
        self.loaded.append(specification)
        return [Prompt("system", "user")]


def make_settings(directory: str) -> HereticSettings:
    return HereticSettings.model_construct(
        model="m",
        init_cache=True,
        init_cache_dir=directory,
    )


class CachedInitReuseTests(unittest.TestCase):
    def test_keyword_rate_reuses_cached_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scorer = KeywordRate(
                heretic_settings=make_settings(directory), settings=Settings()
            )
            cache = scorer.init_cache()
            assert cache is not None
            cache.save({"prompts": [["sys", "usr"]]})

            ctx = _FakeContext(_FakeModelShim())
            scorer.init(cast(Context, ctx))

            self.assertEqual(
                [(p.system, p.user) for p in scorer.prompts], [("sys", "usr")]
            )
            self.assertEqual(ctx.loaded, [])

    def test_kl_divergence_reuses_cached_logprobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scorer = KLDivergence(
                heretic_settings=make_settings(directory), settings=KLSettings()
            )
            cache = scorer.init_cache()
            assert cache is not None
            baseline = torch.zeros(1, 4)
            cache.save({"prompts": [["sys", "usr"]], "baseline_logprobs": baseline})

            ctx = _FakeContext(_FakeModelShim())
            scorer.init(cast(Context, ctx))

            self.assertEqual(len(scorer.prompts), 1)
            self.assertTrue(torch.equal(scorer._baseline_logprobs, baseline))
            self.assertEqual(ctx.loaded, [])

    def test_ara_reuses_cached_module_io_and_applies_lora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            modifier = ARA(
                heretic_settings=make_settings(directory), settings=ARASettings()
            )
            cache = modifier.init_cache()
            assert cache is not None
            cache.save(
                {
                    "good_prompt_count": 5,
                    "bad_prompt_count": 7,
                    "good_module_io": [],
                    "bad_module_io": [],
                }
            )

            model = _FakeModelShim()
            ctx = _FakeContext(model)
            modifier.init(cast(Context, ctx))

            self.assertEqual(modifier.good_prompt_count, 5)
            self.assertEqual(modifier.bad_prompt_count, 7)
            self.assertEqual(model.applied_lora, [modifier.settings.lora_rank])
            self.assertEqual(ctx.loaded, [])


if __name__ == "__main__":
    unittest.main()
