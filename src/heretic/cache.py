# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
On-disk cache for the expensive, base-model part of plugin initialization.

Several plugins do a large amount of deterministic work up front on the *unmodified*
base model: filtering prompt sets by refusal behavior (which generates a response per
prompt), computing KL divergence baseline distributions, and capturing ARA module I/O.
On a large model this initialization can take hours, and it is repeated on every
(resumed) run even though the base model and the relevant settings are unchanged.

This module persists those results keyed by a hash of everything that determines them:
the relevant global settings, the plugin's own settings, and a hash of the Heretic
source code (so editing the code invalidates stale entries automatically). The cached
data is produced from the base model only, so it is never used to evaluate a modified
model.

Because a code edit or a changed setting makes *every* entry a miss, a mismatch is not
treated as final: entries produced under other configurations are discovered and, with
the caller's consent, can be reused anyway (see `InitCache.load_with_fallback`).
"""

import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import questionary
import torch
from pydantic import BaseModel

from .config import Settings

# Bump this whenever the cache format (or the meaning of cached values) changes.
CACHE_FORMAT_VERSION = 1

# Metadata file recording the environment a set of cached entries was produced under,
# so a mismatch can be explained to the user.
ENVIRONMENT_METADATA_FILE = "_environment.json"

# Maximum number of mismatched entries offered for reuse.
_MAX_ALTERNATIVES = 10

_SOURCE_HASH: str | None = None


def _compute_source_hash() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_hash() -> str:
    """
    Hash of the Heretic source tree. Any code change invalidates cached results, which
    keeps a stale cache from silently changing the outcome after an edit.
    """
    global _SOURCE_HASH
    if _SOURCE_HASH is None:
        try:
            _SOURCE_HASH = _compute_source_hash()
        except OSError:
            # Read-only or otherwise inaccessible source tree: fall back to no hash.
            _SOURCE_HASH = ""
    return _SOURCE_HASH


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _vllm_identity(settings: Settings) -> Any:
    """
    Identity of the inference backend that produced the cached base-model results.

    Kept in the environment key so a prompt set or baseline produced by one engine (e.g.
    HuggingFace) is never reused for another (e.g. vLLM loading a different base).
    """
    vllm = getattr(settings, "vllm", None)
    if vllm is None or not vllm.enabled:
        return "disabled"
    return {
        "model": vllm.model or settings.model,
        "mode": vllm.mode,
        "url": vllm.url,
        "dtype": vllm.dtype,
        "max_logprobs": vllm.max_logprobs,
        "logprobs_mode": vllm.logprobs_mode,
    }


def environment_payload(
    settings: Settings, include_backend: bool = True
) -> dict[str, Any]:
    """
    The global settings that determine base-model initialization results, as a plain dict.

    Deliberately excludes settings that do not affect initialization (such as n_trials)
    so that continuing a run with a different trial count does not invalidate the cache.
    `include_backend` controls whether the scoring backend identity is part of the
    payload: scorer results (filtered prompt sets, baselines) are produced by the scoring
    backend, so they include it; modifier results (module I/O) are produced on
    HuggingFace regardless of the backend, so they exclude it and survive a backend
    change.
    """
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source_hash": source_hash(),
        "model": settings.model,
        "model_commit": settings.model_commit,
        "dtypes": settings.dtypes,
        "quantization": str(settings.quantization),
        "attn_implementation": settings.attn_implementation,
        "offload_outputs_to_cpu": settings.offload_outputs_to_cpu,
        "system_prompt": settings.system_prompt,
        "generate_chain_of_thought": settings.generate_chain_of_thought,
        "capture_during_generation": settings.capture_during_generation,
        "max_thought_length": settings.max_thought_length,
        "max_response_length": settings.max_response_length,
        "batch_size": settings.batch_size,
        "capture_batch_size": settings.capture_batch_size,
        "chat_template_kwargs": settings.chat_template_kwargs,
        "chain_of_thought_skips": settings.chain_of_thought_skips,
        "response_prefix": settings.response_prefix,
    }
    if include_backend:
        payload["vllm"] = _vllm_identity(settings)
    return payload


def environment_key(settings: Settings, include_backend: bool = True) -> str:
    """
    Hash of `environment_payload`, used as the per-configuration cache directory name.
    """
    return _hash_json(environment_payload(settings, include_backend))


def plugin_namespace(plugin_class_name: str, plugin_settings: BaseModel | None) -> str:
    """
    Filesystem-safe namespace for a plugin instance, derived from its class name and a
    hash of its settings, so instances with different settings do not collide.
    """
    settings_hash = (
        _hash_json(plugin_settings.model_dump(mode="json"))
        if plugin_settings is not None
        else "0"
    )
    raw = f"{plugin_class_name}-{settings_hash[:16]}"
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in raw)


def _load_cache_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        # Treat an unreadable or outdated entry as a miss.
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _shorten(value: Any, limit: int = 70) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _describe_mtime(path: Path) -> str:
    try:
        modified = path.stat().st_mtime
    except OSError:
        return "age unknown"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(modified))


# One candidate entry for reuse: its environment directory name, its file, and the
# environment payload it recorded (None for entries written before recording existed).
Alternative = tuple[str, Path, dict[str, Any] | None]


class InitCache:
    """
    Managed store for one plugin's initialization results.

    `load()` returns None on a miss (or an unreadable/corrupt entry), in which case the
    caller recomputes and calls `save()`. `load_with_fallback()` additionally offers
    entries for the same plugin that were produced under a different configuration.
    """

    def __init__(
        self,
        directory: str | Path,
        environment: str,
        namespace: str,
        environment_payload: dict[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.environment = environment
        self.namespace = namespace
        self.environment_payload = environment_payload
        self.path = self.directory / environment / f"{namespace}.pt"

    def load(self) -> dict[str, Any] | None:
        return _load_cache_file(self.path)

    def save(self, payload: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file first so an interrupted save cannot leave a
            # partially written cache entry behind.
            temporary = self.path.with_name(self.path.name + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(self.path)

            if self.environment_payload is not None:
                metadata = self.path.parent / ENVIRONMENT_METADATA_FILE
                metadata_temporary = metadata.with_name(metadata.name + ".tmp")
                metadata_temporary.write_text(
                    json.dumps(
                        self.environment_payload,
                        sort_keys=True,
                        default=str,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                metadata_temporary.replace(metadata)
        except OSError:
            # Caching is best effort: never fail initialization because the cache
            # directory is not writable.
            pass

    def find_alternatives(self) -> list[Alternative]:
        """
        Cached entries for the same plugin (same namespace) under a *different*
        environment key, newest first. These were produced under another configuration.
        """
        if not self.directory.is_dir():
            return []

        alternatives: list[Alternative] = []
        for environment_directory in self.directory.iterdir():
            if (
                not environment_directory.is_dir()
                or environment_directory.name == self.environment
            ):
                continue

            path = environment_directory / f"{self.namespace}.pt"
            if not path.is_file():
                continue

            metadata: dict[str, Any] | None = None
            metadata_path = environment_directory / ENVIRONMENT_METADATA_FILE
            if metadata_path.is_file():
                try:
                    metadata = cast(
                        dict[str, Any],
                        json.loads(metadata_path.read_text(encoding="utf-8")),
                    )
                except (OSError, json.JSONDecodeError):
                    metadata = None

            alternatives.append((environment_directory.name, path, metadata))

        alternatives.sort(
            key=lambda alternative: alternative[1].stat().st_mtime,
            reverse=True,
        )
        return alternatives[:_MAX_ALTERNATIVES]

    def describe_differences(self, metadata: dict[str, Any] | None) -> list[str]:
        """
        Human-readable list of the settings that differ between the recorded environment
        of a candidate entry and the current one.
        """
        if metadata is None or self.environment_payload is None:
            return ["differences unknown (entry predates environment recording)"]

        differences: list[str] = []
        for key in sorted(set(metadata) | set(self.environment_payload)):
            old = metadata.get(key, "<absent>")
            new = self.environment_payload.get(key, "<absent>")
            if old == new:
                continue
            if key == "source_hash":
                differences.append("source_hash: differs (Heretic source changed)")
            else:
                differences.append(
                    f"{key}: cached={_shorten(old)} current={_shorten(new)}"
                )
        return differences or ["no differences in the recorded environment"]

    def load_with_fallback(
        self,
        label: str,
        decide: Callable[[list[Alternative]], int | None] | None = None,
    ) -> dict[str, Any] | None:
        """
        Load this plugin's cached initialization, falling back to an entry produced under
        a different configuration if the caller consents.

        On an exact miss, candidate entries for the same plugin are shown with the
        settings that differ, and `decide` returns the index of the one to reuse (or None
        to decline). When `decide` is omitted the user is prompted interactively, and a
        non-interactive session declines.
        """
        payload = self.load()
        if payload is not None:
            return payload

        alternatives = self.find_alternatives()
        if not alternatives:
            return None

        if decide is None:
            decide = self._ask_for_alternative(label)

        index = decide(alternatives)
        if index is None or not 0 <= index < len(alternatives):
            return None
        return _load_cache_file(alternatives[index][1])

    def _ask_for_alternative(
        self, label: str
    ) -> Callable[[list[Alternative]], int | None]:
        def ask(alternatives: list[Alternative]) -> int | None:
            from .utils import print

            print()
            print(
                f"[yellow]No cached initialization for [bold]{label}[/] matches the "
                "current configuration, but entries from a different configuration were "
                "found.[/]"
            )
            for environment, path, metadata in alternatives:
                print(
                    f"* {environment[:12]} ({_describe_mtime(path)})",
                    markup=False,
                )
                for difference in self.describe_differences(metadata):
                    print(f"    {difference}", markup=False)

            if not sys.stdin.isatty():
                print(
                    "[yellow]Not running interactively; ignoring the mismatched "
                    "entries.[/]"
                )
                return None

            if len(alternatives) == 1:
                confirmed = questionary.confirm(
                    "Reuse this cached initialization despite the differences?",
                    default=False,
                ).ask()
                return 0 if confirmed else None

            choices = [
                questionary.Choice(
                    title=f"{environment[:12]} ({_describe_mtime(path)})",
                    value=index,
                )
                for index, (environment, path, _) in enumerate(alternatives)
            ]
            choices.append(
                questionary.Choice(title="Do not reuse any cached entry", value=None)
            )
            return cast(
                int | None,
                questionary.select("Which cached entry?", choices=choices).ask(),
            )

        return ask
