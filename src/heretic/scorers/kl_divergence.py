# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Literal

import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer, apply_refusal_filter
from heretic.utils import REFUSAL_MARKERS, format_dataset_specification, print


class Settings(BaseModel):
    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts used to measure KL divergence from original model.",
    )

    capture_position: Literal["auto", "first_token", "first_answer_token"] = Field(
        default="auto",
        description=(
            'Which next-token distribution to compare against the baseline: '
            '"first_token" (start of the response, i.e. the start of reasoning), '
            '"first_answer_token" (the first token after the reasoning block closes), '
            'or "auto" (follow the global generate_chain_of_thought setting).'
        ),
    )

    filter_prompts: Literal["none", "keep_refused", "drop_refused"] = Field(
        default="none",
        description=(
            "Whether to filter the prompt set by the base model's refusal behavior before "
            'evaluating: "keep_refused" keeps only prompts the model refuses, '
            '"drop_refused" drops them, "none" disables filtering. Uses refusal_markers.'
        ),
    )

    refusal_markers: list[str] = Field(
        default=REFUSAL_MARKERS,
        description="Strings used to detect refusals when filter_prompts is enabled.",
    )


class KLDivergence(Scorer):
    """
    KL divergence between current model and baseline.

    Measures how much the model's behavior has drifted from baseline.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "KL divergence"

    @property
    def _after_reasoning(self) -> bool | None:
        if self.settings.capture_position == "first_token":
            return False
        if self.settings.capture_position == "first_answer_token":
            return True
        return None

    def init(self, ctx: Context) -> None:
        print()
        print(
            f"Loading KL divergence evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        self.prompts = apply_refusal_filter(
            ctx,
            self.prompts,
            self.settings.filter_prompts,
            self.settings.refusal_markers,
        )

        print("* Obtaining baseline probability distributions...")
        baseline_logits = ctx.get_logits(self.prompts, self._after_reasoning)

        self._baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

    def get_score(self, ctx: Context) -> Score:
        logits = ctx.get_logits(self.prompts, self._after_reasoning)
        logprobs = F.log_softmax(logits, dim=-1)

        kl_divergence = F.kl_div(
            logprobs,
            self._baseline_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()

        return Score(
            value=kl_divergence,
            rich_display=f"[bold]{kl_divergence:.4f}[/]",
            md_display=f"{kl_divergence:.4f}",
        )

    def get_baseline_score(self, ctx: Context) -> Score:
        return Score(
            value=0,
            rich_display="[bold]0[/] [italic](by definition)[/]",
            md_display="0 *(by definition)*",
        )
