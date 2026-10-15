# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import importlib
import importlib.util
import inspect
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import (
    Annotated,
    Any,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor

from .cache import InitCache, environment_key, plugin_namespace
from .config import DatasetSpecification, SingleDatasetSpecification
from .config import Settings as HereticSettings
from .inference import HFBackend, InferenceBackend
from .model import (
    CaptureRequest,
    GenerationRecord,
    Model,
    ModuleIO,
    merge_module_io_rows,
    split_module_io_rows,
)
from .utils import Prompt, batchify, deep_merge_dicts, is_keyword_match, load_prompts

T = TypeVar("T")


def get_plugin_namespace(
    model_extra: dict[str, Any] | None,
    namespace: str,
) -> dict[str, Any]:
    """
    Returns the config dict from the `[<namespace>]` TOML table.
    """
    cur: Any = model_extra
    for part in namespace.split("."):
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(part)

    if cur is None:
        return {}
    if not isinstance(cur, dict):
        raise TypeError(
            f"Plugin namespace [{namespace}] must be a table/object, got {type(cur).__name__}"
        )
    return cur


def is_builtin_plugin(name: str) -> bool:
    """
    Whether the plugin name refers to a plugin that ships with Heretic.

    Only built-in plugins can be resolved when reproducing a model, so external
    plugins (file paths or third-party import paths) disable the reproducibility
    offer during upload.
    """
    return name.startswith("heretic.")


def load_plugin(
    name: str,
    base_class: type[T],
) -> type[T]:
    """
    Load a plugin class from either a filesystem `.py` file or a fully-qualified Python import path.
    Also checks that the class exists in the module and that it
    subclasses the correct Plugin subclass (e.g Scorer).

    Accepted forms:
    - `path/to/plugin.py:MyPluginClass` (relative or absolute): load `MyPluginClass`
      from that file.
    - `fully.qualified.module.MyPluginClass`: import the module and load the class.
    """

    def validate_class(module: ModuleType, class_name: str) -> type[Any]:
        """
        Checks that the module actually exports the class as claimed and returns the class.
        """
        obj = getattr(module, class_name, None)
        if not inspect.isclass(obj):
            raise ValueError(
                f"Plugin '{name}' does not export a class named '{class_name}'"
            )
        return obj

    # Common user trap with filepath imports.
    if name.endswith(".py"):
        raise ValueError(
            "You must append the plugin class name to the filepath like this: path/to/plugin.py:ClassName"
        )

    # File path with explicit class name, e.g. "C:\\path\\plugin.py:MyPlugin".
    if ":" in name:
        file_path, class_name = name.rsplit(":", 1)
        if not file_path.endswith(".py") or not class_name:
            raise ValueError(
                "File-based plugin must use the form 'path/to/plugin.py:ClassName'"
            )

        plugin_path = Path(file_path)
        if not plugin_path.is_absolute():
            plugin_path = Path.cwd() / plugin_path
        plugin_path = plugin_path.resolve()

        if not plugin_path.is_file():
            raise ImportError(f"Plugin file '{plugin_path}' does not exist")

        # We're writing directly to the sys.modules dict,
        # so the typical restrictions on module names
        # (no dots, slashes, etc.) don't apply.
        module_name = f"heretic_plugin_{plugin_path}"

        # Reuse already-loaded modules to avoid re-executing the plugin on repeated loads.
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not load plugin '{name}' (invalid module spec)"
                )

            module = importlib.util.module_from_spec(spec)

            # Cache before executing to match normal import semantics and allow
            # circular imports. If execution fails, remove the entry.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise

        plugin_cls = validate_class(module, class_name)
    # Fully-qualified import path, e.g "heretic.scorers.keyword_rate.KeywordRate".
    else:
        if "." not in name:
            raise ValueError(
                "Import-based plugin must use the form 'fully.qualified.module.ClassName'"
            )
        module_name, class_name = name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Error loading plugin '{name}': {e}") from e
        plugin_cls = validate_class(module, class_name)

    if not issubclass(plugin_cls, base_class):
        raise TypeError(f"Plugin '{name}' must subclass {base_class.__name__}")

    return plugin_cls


class Context:
    """
    Runtime context passed to plugins.
    Acts as a quasi-API for plugins to access Heretic functionality.
    """

    def __init__(
        self,
        settings: HereticSettings,
        model: Model,
        backend: InferenceBackend | None = None,
    ) -> None:
        self._model = model
        self._settings = settings
        # Which engine serves generation. Defaults to the in-process HuggingFace model;
        # the evaluator injects a shared vLLM backend for pure-inference scoring.
        self._backend: InferenceBackend = (
            backend if backend is not None else HFBackend(model)
        )
        # Per-prompt generation records, keyed by prompt, so that refusal scoring and
        # logit capture share a single generation instead of regenerating the reasoning.
        self._generation_records: dict[tuple[str, str], GenerationRecord] = {}
        # Whether to capture logits/module I/O during generation for prompts generated in
        # this context (only acted on when capture_during_generation is enabled). Logits
        # are captured for every generation because KL scorers need them; module I/O is
        # only needed by the modifier, which enables it before generating its prompts.
        self.capture_logits_default = getattr(
            settings, "capture_during_generation", False
        )
        self.capture_module_io_default = False

    def _prompt_key(self, prompt: Prompt) -> tuple[str, str]:
        return (prompt.system, prompt.user)

    def get_generation_records(self, prompts: list[Prompt]) -> list[GenerationRecord]:
        """
        Return one generation record per prompt, generating (and caching) only the
        prompts that have not been generated in this context yet.
        """
        missing: list[Prompt] = []
        seen: set[tuple[str, str]] = set()
        for prompt in prompts:
            key = self._prompt_key(prompt)
            if key not in self._generation_records and key not in seen:
                seen.add(key)
                missing.append(prompt)

        if missing:
            records = self._backend.get_generation_records(
                missing,
                capture=CaptureRequest(
                    logits=self.capture_logits_default,
                    module_io=self.capture_module_io_default,
                ),
            )
            for prompt, record in zip(missing, records):
                self._generation_records[self._prompt_key(prompt)] = record

        return [
            self._generation_records[self._prompt_key(prompt)] for prompt in prompts
        ]

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        """Get model responses (generated once per prompt within this context)."""
        return [record.response for record in self.get_generation_records(prompts)]

    def filter_prompts_by_refusal(
        self,
        prompts: list[Prompt],
        *,
        keep_refused: bool,
        refusal_markers: list[str],
    ) -> list[Prompt]:
        """
        Filter prompts by the model's refusal behavior.

        `keep_refused=True` keeps prompts the model refuses (for sets that are meant to
        be refused, e.g. harmful prompts); `keep_refused=False` drops them (for sets that
        are meant to be answered, e.g. harmless prompts). May return an empty list; the
        caller decides how to handle that.
        """
        responses = self.get_responses(prompts)
        return [
            prompt
            for prompt, response in zip(prompts, responses)
            if is_keyword_match(response, refusal_markers) == keep_refused
        ]

    def get_logits(
        self,
        prompts: list[Prompt],
        after_reasoning: bool | None = None,
    ) -> Tensor:
        if after_reasoning is None:
            after_reasoning = self._settings.generate_chain_of_thought

        # Reuse the reasoning already generated for these prompts (e.g. for refusal
        # filtering) instead of regenerating it to locate the capture position. When the
        # capture was taken during that generation, use it directly.
        if after_reasoning:
            # First answer token: the token after the reasoning block closes.
            records = self.get_generation_records(prompts)
            captured = [record.answer_logits for record in records]
            capture_ids = [record.capture_ids for record in records]
        elif self._settings.generate_chain_of_thought:
            # First response token: skip the model's thinking-start marker so that the
            # captured distribution is over the first token that carries content, rather
            # than the (near-deterministic) marker token.
            records = self.get_generation_records(prompts)
            captured = [record.content_logits for record in records]
            capture_ids = [record.first_content_ids for record in records]
        else:
            return self._model.get_logits_batched(prompts, after_reasoning)

        return self._assemble_logits(captured, capture_ids)

    def get_logprobs(
        self,
        prompts: list[Prompt],
        after_reasoning: bool | None = None,
    ) -> Tensor:
        """
        Log-probabilities at the capture position.

        This is the log-probability counterpart of `get_logits` and the contract KL
        scorers use, because a backend that returns log-probabilities but not raw logits
        (e.g. vLLM) can serve it. Logits captured during generation are converted to
        log-probabilities; rows without a capture are fetched from the backend.
        """
        if after_reasoning is None:
            after_reasoning = self._settings.generate_chain_of_thought

        if after_reasoning:
            records = self.get_generation_records(prompts)
            captured = [self._to_logprobs(record.answer_logits) for record in records]
            capture_ids = [record.capture_ids for record in records]
        elif self._settings.generate_chain_of_thought:
            records = self.get_generation_records(prompts)
            captured = [self._to_logprobs(record.content_logits) for record in records]
            capture_ids = [record.first_content_ids for record in records]
        else:
            return self._backend.get_logprobs_prompt(prompts)

        return self._assemble_logprobs(captured, capture_ids)

    @staticmethod
    def _to_logprobs(logits: Tensor | None) -> Tensor | None:
        if logits is None:
            return None
        return F.log_softmax(logits, dim=-1)

    def _assemble_logprobs(
        self,
        captured: list[Tensor | None],
        capture_ids: list[list[int]],
    ) -> Tensor:
        return self._assemble_rows(
            captured, capture_ids, self._backend.get_logprobs_from_capture_ids
        )

    def _assemble_logits(
        self,
        captured: list[Tensor | None],
        capture_ids: list[list[int]],
    ) -> Tensor:
        """
        Combine per-prompt logits captured during generation with a separate capture
        forward for any prompts that were not captured (e.g. reasoning that never
        closed). When nothing was captured this is exactly the old capture-forward path.
        """
        return self._assemble_rows(
            captured, capture_ids, self._model.get_logits_from_capture_ids
        )

    def _assemble_rows(
        self,
        captured: list[Tensor | None],
        capture_ids: list[list[int]],
        fetch: Callable[[list[list[int]]], Tensor],
    ) -> Tensor:
        """
        Combine per-prompt rows captured during generation with rows fetched in a
        separate pass (`fetch`) for prompts that were not captured. When nothing was
        captured this is exactly the separate-pass path.
        """
        missing = [index for index, value in enumerate(captured) if value is None]
        if not missing:
            return torch.stack(cast(list[Tensor], captured), dim=0)

        fallback = torch.cat(
            [
                fetch(batch)
                for batch in batchify(
                    [capture_ids[index] for index in missing],
                    self._settings.capture_batch_size,
                )
            ],
            dim=0,
        )

        if len(missing) == len(captured):
            return fallback

        result: list[Tensor | None] = list(captured)
        for offset, index in enumerate(missing):
            result[index] = fallback[offset]

        reference = next(value for value in result if value is not None)
        return torch.stack(
            [cast(Tensor, value).to(reference) for value in result],
            dim=0,
        )

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        return self._model.get_residuals_batched(prompts)

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        """
        Mean residuals at the capture position. Reuses the reasoning already generated
        for these prompts within this context instead of regenerating it.
        """
        if not self._settings.generate_chain_of_thought:
            return self._model.get_residuals_mean(
                prompts,
                winsorization_quantile=winsorization_quantile,
            )

        capture_ids = [
            record.capture_ids for record in self.get_generation_records(prompts)
        ]
        return self._model.get_residuals_mean(
            prompts,
            winsorization_quantile=winsorization_quantile,
            capture_ids=capture_ids,
        )

    def get_module_io(self, prompts: list[Prompt]) -> ModuleIO:
        """
        Module I/O at the capture position. Reuses the reasoning already generated for
        these prompts within this context instead of regenerating it, and uses module I/O
        captured during that generation when available.
        """
        if not self._settings.generate_chain_of_thought:
            return self._model.get_module_io_batched(prompts)

        records = self.get_generation_records(prompts)
        rows = [record.module_io_row for record in records]
        missing = [index for index, row in enumerate(rows) if row is None]

        if not missing:
            return merge_module_io_rows(
                cast(list[ModuleIO], rows), len(self._model.get_layers())
            )

        capture_ids = [records[index].capture_ids for index in missing]

        if len(missing) == len(rows):
            # Nothing was captured: exactly the old batched capture-forward path.
            return self._model.get_module_io_batched(prompts, capture_ids)

        fallback_io = self._model.get_module_io_batched(
            [prompts[index] for index in missing],
            capture_ids,
        )
        fallback_rows = split_module_io_rows(fallback_io, len(missing))
        for offset, index in enumerate(missing):
            rows[index] = fallback_rows[offset]

        return merge_module_io_rows(
            cast(list[ModuleIO], rows), len(self._model.get_layers())
        )

    def get_model(self) -> Model:
        """
        Prefer managed methods (`get_responses` etc.) unless you
        actually need access to the model object.
        """
        return self._model

    def load_prompts(self, specification: DatasetSpecification) -> list[Prompt]:
        return load_prompts(self._settings, specification)


class Plugin:
    """
    Base class for Heretic plugins.

    Plugins may define:
    - `settings: <BaseModelSubclass>` type annotation (recommended)
      Heretic will validate the corresponding config table against it and pass
      an instance as `settings`.
    """

    @property
    def reproducible(self) -> bool:
        """
        Whether runs using this plugin can be reproduced bit-for-bit.

        Set to False when the plugin's behavior is not deterministic or depends on
        state outside the pinned config, for example:
        - It calls an external service (e.g. an LLM judge over the OpenAI API).
        - It reads credentials or config from the environment (env vars, files).
        - It is otherwise non-deterministic (network, wall-clock, unseeded RNG).

        Defaults to False; override to True in your plugin class if any of the
        above DO NOT apply.
        """
        return False

    def __init__(
        self,
        *,
        heretic_settings: HereticSettings,
        settings: BaseModel | None = None,
    ) -> None:
        # Plugins that declare a settings schema should always receive
        # validated plugin settings from the evaluator.
        settings_model = self.__class__.get_settings_model()
        if settings_model is not None:
            if settings is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires settings to be validated"
                )
            if not isinstance(settings, settings_model):
                raise TypeError(
                    f"{self.__class__.__name__}.settings must be an instance of "
                    f"{settings_model.__name__}"
                )
        self.settings = settings
        self.heretic_settings = heretic_settings

    @classmethod
    def validate_contract(cls) -> None:
        """
        Validate the plugin contract.

        - Plugins must not define a constructor (`__init__`). Initialization is
          handled by `Plugin.__init__` and an optional `init(ctx)` method.
        - Plugin subclasses may define `settings: <BaseModelSubclass>` to declare a settings schema.
        """
        if "__init__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not define __init__(). "
                "Use an optional init(ctx) method for plugin-specific initialization."
            )

    @classmethod
    def get_settings_model(cls) -> type[BaseModel] | None:
        """
        Return the plugin settings model, if present.
        - If the plugin has a `settings: <BaseModelSubclass>` type annotation,
          that type is used as the settings schema.
        - Otherwise: no settings schema.
        """

        def unwrap_settings_type(tp: Any) -> Any:
            """Unwrap `Annotated[T, ...]`."""
            while True:
                origin = get_origin(tp)
                if origin is Annotated:
                    tp = get_args(tp)[0]
                    continue
                return tp

        hints = get_type_hints(cls, include_extras=True)
        annotated = hints.get("settings")
        if annotated is None:
            return None

        model = unwrap_settings_type(annotated)
        origin = get_origin(model)
        if origin in (Union, types.UnionType) and type(None) in get_args(model):
            raise TypeError(
                f"{cls.__name__}.settings must not be Optional; "
                "use a non-optional pydantic.BaseModel subclass (e.g. `settings: Settings`)."
            )
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise TypeError(
                f"{cls.__name__}.settings must be annotated with a pydantic.BaseModel subclass"
            )
        return model

    @classmethod
    def get_settings_raw(
        cls,
        model_extra: dict[str, Any] | None,
        top_namespace: str,
        instance_name: str | None,
    ) -> dict[str, Any]:
        """
        Build the raw settings dict for a plugin class and optional instance.

        Config rules:
        - Base settings live in `[<top_namespace>.ClassName]` (applies to all instances).
        - Instance overrides live in `[<top_namespace>.ClassName_<instance_name>]` (preferred).
        - Only merge/validate keys that exist in the plugin Settings schema.
        """
        settings_model = cls.get_settings_model()
        if settings_model is None:
            # No settings schema: nothing to merge/validate.
            return {}

        class_name = cls.__name__

        namespaces = [f"{top_namespace}.{class_name}"]
        if instance_name:
            namespaces.append(f"{top_namespace}.{class_name}_{instance_name}")

        merged_settings: dict[str, Any] = {}
        allowed_keys = set(settings_model.model_fields.keys())

        for namespace in namespaces:
            raw_table = get_plugin_namespace(model_extra, namespace)
            filtered = {k: v for k, v in raw_table.items() if k in allowed_keys}
            merged_settings = deep_merge_dicts(merged_settings, filtered)

        return merged_settings

    @classmethod
    def validate_settings(
        cls,
        raw_namespace: dict[str, Any] | None,
    ) -> BaseModel | None:
        """
        Validates plugin settings for this plugin class.

        - If a settings model is present: returns an instance of that model.
        - Otherwise returns None.
        """
        settings_model = cls.get_settings_model()
        if settings_model is None:
            return None
        return settings_model.model_validate(raw_namespace or {})

    def get_dataset_specifications(self) -> list[DatasetSpecification]:
        """
        Collect the dataset specifications declared in the settings
        of the plugin.
        """
        if self.settings is None:
            return []
        specifications = []
        for value in dict(self.settings).values():
            if isinstance(value, SingleDatasetSpecification) or (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], SingleDatasetSpecification)
            ):
                specifications.append(value)
        return specifications

    def init_cache(self) -> InitCache | None:
        """
        Cache handle for this plugin's base-model initialization results, or None when
        caching is disabled (`init_cache = false`). Cached values are keyed by the global
        settings that determine them, the plugin's own settings, and the source hash, so
        they are reused across runs but invalidated when anything relevant changes.
        """
        if not self.heretic_settings.init_cache:
            return None
        return InitCache(
            directory=self.heretic_settings.init_cache_dir,
            environment=environment_key(
                self.heretic_settings,
                include_backend=self.backend_dependent_init,
            ),
            namespace=plugin_namespace(self.__class__.__name__, self.settings),
        )

    @property
    def backend_dependent_init(self) -> bool:
        """
        Whether this plugin's cached initialization results depend on the scoring backend.

        Defaults to False because modifiers compute their initialization on HuggingFace
        regardless of the backend. Scorers, whose filtered prompt sets and baselines are
        produced by the scoring backend, override this to True.
        """
        return False

    def init(self, ctx: Context) -> None:
        """
        Runs before the plugin's main functionality.

        Override this in subclasses to do one-time setup (e.g. load prompts, compute
        baselines).
        """
        return
