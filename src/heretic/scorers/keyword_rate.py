# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from typing import Literal

from pydantic import BaseModel, Field
from rich.markup import escape

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.scorer import Context, Score, Scorer, apply_refusal_filter
from heretic.utils import (
    REFUSAL_MARKERS,
    Prompt,
    format_dataset_specification,
    is_keyword_match,
    print,
)


class Settings(BaseModel):
    score_name: str = Field(
        default="Refusals",
        description="Name that describes what the configured keyword rate measures.",
    )

    prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="test[:100]",
            column="text",
        ),
        description="Dataset of prompts to evaluate the keyword match rate on.",
    )

    keyword_markers: list[str] = Field(
        default=REFUSAL_MARKERS,
        description="Strings whose presence in a response (case insensitive) identifies the response as a keyword match.",
    )

    filter_prompts: Literal["none", "keep_refused", "drop_refused"] = Field(
        default="none",
        description=(
            "Whether to filter the prompt set by the base model's refusal behavior before "
            'evaluating: "keep_refused" keeps only prompts the model refuses, '
            '"drop_refused" drops them, "none" disables filtering. Uses keyword_markers.'
        ),
    )

    print_responses: bool = Field(
        default=False,
        description="Whether to print prompt/response pairs when counting keyword matches.",
    )


class KeywordRate(Scorer):
    """
    Counts responses containing any of a list of keyword markers.

    Markers are defined in the [scorer.KeywordRate] config section.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def requires_model_internals(self) -> bool:
        return False

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        cache = self.init_cache()

        if cache is not None:
            cached = cache.load_with_fallback(self.score_name)
            if cached is not None:
                self.prompts = [
                    Prompt(system=system, user=user)
                    for system, user in cached["prompts"]
                ]
                print()
                print(
                    f"Reusing cached initialization for [bold]{self.score_name}[/] "
                    f"([bold]{len(self.prompts)}[/] prompts)."
                )
                return

        print()
        print(
            f"Loading {self.settings.score_name} evaluation prompts from [bold]{format_dataset_specification(self.settings.prompts)}[/]..."
        )
        self.prompts = ctx.load_prompts(self.settings.prompts)
        print(f"* [bold]{len(self.prompts)}[/] prompts loaded")

        self.prompts = apply_refusal_filter(
            ctx,
            self.prompts,
            self.settings.filter_prompts,
            self.settings.keyword_markers,
        )

        if cache is not None:
            cache.save({"prompts": [[p.system, p.user] for p in self.prompts]})

    def get_score(self, ctx: Context) -> Score:
        match_count = 0
        responses = ctx.get_responses(self.prompts)
        for prompt, response in zip(self.prompts, responses):
            is_match = self._is_match(response)
            if is_match:
                match_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {escape(prompt.system)}")
                print(f"[bold]Prompt:[/] {escape(prompt.user)}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                else:
                    response = escape(response)
                print(
                    f"[bold]Response:[/] [{'red' if is_match else 'green'}]{response}[/]"
                )

        if self.settings.print_responses:
            print()

        return Score(
            value=float(match_count / len(self.prompts)),
            rich_display=f"[bold]{match_count}[/]/{len(self.prompts)}",
            md_display=f"{match_count}/{len(self.prompts)}",
        )

    def _is_match(self, response: str) -> bool:
        return is_keyword_match(response, self.settings.keyword_markers)
