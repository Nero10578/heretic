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
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel

from .config import Settings

# Bump this whenever the cache format (or the meaning of cached values) changes.
CACHE_FORMAT_VERSION = 1

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


def environment_key(settings: Settings) -> str:
    """
    Hash of the global settings that determine the base-model initialization results.
    Deliberately excludes settings that do not affect initialization (such as n_trials)
    so that continuing a run with a different trial count does not invalidate the cache.
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
    return _hash_json(payload)


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


class InitCache:
    """
    JSON/managed store for one plugin's initialization results.

    `load()` returns None on a miss (or an unreadable/corrupt entry), in which case the
    caller recomputes and calls `save()`.
    """

    def __init__(self, directory: str | Path, environment: str, namespace: str) -> None:
        self.directory = Path(directory)
        self.path = self.directory / environment / f"{namespace}.pt"

    def load(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            payload = torch.load(
                self.path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception:
            # Treat an unreadable or outdated entry as a miss.
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary file first so an interrupted save cannot leave a
            # partially written cache entry behind.
            temporary = self.path.with_name(self.path.name + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(self.path)
        except OSError:
            # Caching is best effort: never fail initialization because the cache
            # directory is not writable.
            pass
