# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from .config import Settings as HereticSettings
from .plugin import Context, Plugin
from .utils import Prompt, print


@dataclass
class Score:
    """
    Result of evaluating a scorer.

    - `value`: scalar value used for optimization (if enabled).
    - `rich_display`: formatted Rich markup shown to the user in logs/console.
    - `md_display`: formatted value in the HF model card.
    """

    value: float
    rich_display: str
    md_display: str


def apply_refusal_filter(
    ctx: Context,
    prompts: list[Prompt],
    mode: str,
    refusal_markers: list[str],
) -> list[Prompt]:
    """
    Apply a refusal-based prompt filter to a scorer's prompt set.

    `mode` is "none" (no filtering), "keep_refused" (keep only prompts the base model
    refuses), or "drop_refused" (drop prompts the base model refuses). Falls back to the
    unfiltered set if filtering would remove every prompt.
    """
    if mode == "none" or not prompts:
        return prompts

    filtered = ctx.filter_prompts_by_refusal(
        prompts,
        keep_refused=(mode == "keep_refused"),
        refusal_markers=refusal_markers,
    )

    if not filtered:
        print(
            "[yellow]Warning: Prompt filtering removed all evaluation prompts. "
            "Using the unfiltered set.[/]"
        )
        return prompts

    print(f"* Filtered to [bold]{len(filtered)}[/]/{len(prompts)} prompts")
    return filtered


class Scorer(Plugin, ABC):
    """
    Abstract base class for scorer plugins.

    Scorers evaluate model behavior and return a Score.

    Examples: Counting refusals, measuring KL divergence, etc.
    """

    @property
    def score_name(self) -> str:
        """
        The name of the `Score` object returned by `get_score()`.
        This is what shows up in the CLI and Markdown metrics on HF.
        """
        return self.__class__.__name__

    @property
    def requires_model_internals(self) -> bool:
        """
        Whether the scorer needs model internals that only the in-process HuggingFace
        backend exposes (residuals, module I/O, or direct model access).

        Scorers that use only `get_responses` and `get_logprobs` can be served by the
        vLLM backend and should return False. Defaults to True, so an unknown scorer
        stays on the HuggingFace path.
        """
        return True

    def __init__(
        self,
        heretic_settings: HereticSettings,
        settings: BaseModel | None = None,
    ) -> None:
        super().__init__(heretic_settings=heretic_settings, settings=settings)

    @abstractmethod
    def get_score(self, ctx: Context) -> Score:
        """
        Return a `Score` given the evaluation context.
        The `value` of the `Score` must be of the order of magnitude 1
        to ensure that all scores are comparable during co-optimization.
        """

    def get_baseline_score(self, ctx: Context) -> Score:
        """
        Calculates a baseline score.

        Defaults to the current `get_score(...)` implementation and can be
        overridden by scorers that need a distinct baseline.
        """
        return self.get_score(ctx)
