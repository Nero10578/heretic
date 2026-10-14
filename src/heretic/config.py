# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# !!!IMPORTANT!!!
#
# Any settings added to the classes defined in this module
# must be evaluated for privacy implications and have
# exclude=True set in their field definitions if appropriate.


class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class SingleDatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the dataset.",
    )

    config: str | None = Field(
        default=None,
        description=(
            "Dataset config/subset name. Each config can have its own split. "
            "Used to load a specific config of a dataset that has multiple configurations."
        ),
    )

    split: str | None = Field(
        default=None,
        description="Portion of the dataset to use. Required for datasets, optional for plain text files.",
    )

    column: str | None = Field(
        default=None,
        description="Column in the dataset that contains the prompts. Required for datasets, ignored for plain text files.",
    )

    prefix: str = Field(
        default="",
        description="Text to prepend to each prompt.",
    )

    suffix: str = Field(
        default="",
        description="Text to append to each prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use with the prompts (overrides global system prompt if set).",
    )


DatasetSpecification: TypeAlias = (
    SingleDatasetSpecification | list[SingleDatasetSpecification]
)


class ScorerConfig(BaseModel):
    """
    Configuration for a scorer plugin.

    TOML format:
    - { plugin = "<plugin>", optimization = "<optimization>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    optimization: Literal["minimize", "maximize", "none"] = Field(
        description=(
            "Optimization direction for this scorer. "
            '"minimize" / "maximize" to include the scorer as an objective, '
            '"none" to compute the score without optimizing for it.'
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[scorer.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class ModifierConfig(BaseModel):
    """
    Configuration for a modifier plugin.

    TOML format:
    - { plugin = "<plugin>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[modifier.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class BenchmarkSpecification(BaseModel):
    task: str = Field(
        description="Task ID of the benchmark in the Language Model Evaluation Harness."
    )

    name: str = Field(description="Name of the benchmark for presentation purposes.")

    description: str = Field(
        description="Description of the benchmark for presentation purposes."
    )


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")

    model_commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the model.",
    )

    evaluate_model: str | None = Field(
        default=None,
        description=(
            "If this model ID or path is set, then instead of abliterating the main model, "
            "evaluate this model relative to the main model."
        ),
        exclude=True,
    )

    collect_reproducibles: str | None = Field(
        default=None,
        description=(
            "If this directory path is set, then instead of abliterating a model, "
            "download all reproduce.json files from public Heretic model repositories "
            "on Hugging Face, and store them in that directory for archival purposes."
        ),
        exclude=True,
    )

    reproduce: str | None = Field(
        default=None,
        description=(
            "If this path or URL to a reproduce.json file is set, load reproduction information "
            "from that file, and attempt to reproduce the abliterated model it originated from."
        ),
        exclude=True,
    )

    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If that doesn't work (e.g. on pre-Ampere hardware), fall back to float16.
            "float16",
            # If "auto" resolves to float32, and that fails because it is too large,
            # and float16 fails due to range issues, try bfloat16.
            "bfloat16",
            # If neither of those work, fall back to float32 (which will of course fail
            # if that was the dtype "auto" resolved to).
            "float32",
        ],
        description=(
            "List of PyTorch dtypes to try when loading model tensors. "
            "If loading with a dtype fails, the next dtype in the list will be tried."
        ),
    )

    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description=(
            "Quantization method to use when loading the model. Options: "
            '"none" (no quantization), '
            '"bnb_4bit" (4-bit quantization using bitsandbytes).'
        ),
    )

    device_map: str | dict[str, int | str] = Field(
        default="auto",
        description="Device map to pass to Accelerate when loading the model.",
    )

    max_memory: dict[str, str] | None = Field(
        default=None,
        description='Maximum memory to allocate per device (e.g., { "0" = "20GB", "cpu" = "64GB" }).',
    )

    attn_implementation: Literal[
        "auto",
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
        "sdpa",
        "eager",
    ] = Field(
        default="auto",
        description=(
            'Attention implementation for inference, passed to from_pretrained. "auto" '
            "selects FlashAttention when the flash-attn package is installed, which is "
            "much faster for the long reasoning sequences (otherwise the library default, "
            "usually SDPA, is used). FlashAttention is not bit-exact with SDPA, so it "
            "changes results slightly and invalidates the reproducibility hashes."
        ),
    )

    offload_outputs_to_cpu: bool = Field(
        default=True,
        description=(
            "Whether to move intermediate analysis tensors (such as residuals and logprobs) "
            "to CPU memory as soon as possible to reduce peak VRAM usage. "
            "This lowers peak VRAM usage during residual analysis and evaluation, "
            "but may slightly reduce performance due to host/device transfers."
        ),
    )

    batch_size: NonNegativeInt = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    capture_batch_size: PositiveInt = Field(
        default=8,
        description=(
            "Batch size for the capture forwards (KL logits, residuals, module I/O). "
            "These run one forward over the whole prompt plus reasoning, so their peak "
            "VRAM and kernel size scale with the batch times the sequence length. A "
            "smaller value than batch_size avoids out-of-memory errors and driver "
            "launch timeouts on long reasoning, at little throughput cost."
        ),
    )

    max_batch_size: PositiveInt = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    batch_size_test_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:256]",
            column="text",
        ),
        description="Dataset of prompts to use for automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    max_response_length: PositiveInt = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    generate_chain_of_thought: bool = Field(
        default=False,
        description=(
            "If true, let the model generate its reasoning naturally (as dictated by its "
            "chat template) instead of pre-filling a closed reasoning block. Refusals are "
            "then detected only in the text after the reasoning completes, and residuals, "
            "module I/O, and KL logits are captured at the first answer token. Generation "
            "is capped by max_thought_length + max_response_length."
        ),
    )

    max_thought_length: PositiveInt = Field(
        default=2048,
        description=(
            "Maximum number of tokens to generate for the reasoning phase when "
            "generate_chain_of_thought is true. This budget is separate from "
            "max_response_length, which bounds only the final answer."
        ),
    )

    abort_repeating_generation: bool = Field(
        default=True,
        description=(
            "If true, stop generating for a prompt as soon as its output falls into a "
            "repetition loop (the last repetition_min_repeats cycles of length up to "
            "repetition_period_max are identical). Broken models otherwise burn the full "
            "token budget emitting gibberish. Truncated responses are treated as refusals."
        ),
    )

    repetition_period_max: PositiveInt = Field(
        default=16,
        description=(
            "Maximum length of the repeated cycle used for repetition-loop detection."
        ),
    )

    repetition_min_repeats: PositiveInt = Field(
        default=12,
        description=(
            "How many identical consecutive cycles of length up to repetition_period_max "
            "must appear before a repetition loop is detected."
        ),
    )

    abort_broken_trials: bool = Field(
        default=False,
        description=(
            "If true, prune the whole trial when at least abort_broken_trial_fraction of "
            "a generation's prompts were stopped by repetition-loop detection. Pruning "
            "saves the rest of the trial's evaluation but records no score, so the "
            "optimizer gets no signal for that region. By default the trial is scored "
            "instead: a looping model refuses everything (empty answers count as "
            "refusals), so it is already the worst result and steers the search away."
        ),
    )

    abort_broken_trial_fraction: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of repetition-stopped prompts above which the trial is pruned."
        ),
    )

    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional keyword arguments passed to the tokenizer's apply_chat_template. "
            "Use this for template options such as enable_thinking = true."
        ),
    )

    response_prefix: str | None = Field(
        default=None,
        description=(
            "Common prefix to assume for all responses, so that evaluation happens "
            "at the point where responses start to differ for different prompts. "
            "If not set, the prefix is determined automatically by comparing multiple responses."
        ),
    )

    response_prefix_test_prompts: DatasetSpecification = Field(
        default=[
            SingleDatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="train[:100]",
                column="text",
            ),
            SingleDatasetSpecification(
                dataset="mlabonne/harmful_behaviors",
                split="train[:100]",
                column="text",
            ),
        ],
        description="Dataset of prompts to use for automatically determining the response prefix.",
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    chain_of_thought_skips: list[tuple[str, str]] = Field(
        default=[
            # Most thinking models.
            (
                "<think>",
                "<think></think>",
            ),
            # gpt-oss.
            (
                "<|channel|>analysis<|message|>",
                "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
            ),
            # Unknown, suggested by user.
            (
                "<thought>",
                "<thought></thought>",
            ),
            # Unknown, suggested by user.
            (
                "[THINK]",
                "[THINK][/THINK]",
            ),
            # Gemma 4 channel-based reasoning. Requires
            # `chat_template_kwargs = { enable_thinking = true }`.
            (
                "<|channel>thought\n",
                "<|channel>thought\n<channel|>",
            ),
        ],
        description=(
            "List of pairs of the form (cot_initializer, closed_cot_block) used to skip "
            "the Chain-of-Thought block in responses, so that evaluation happens "
            "at the start of the actual response."
        ),
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    print_debug_information: bool = Field(
        default=False,
        description="Whether to print additional information that can help with debugging.",
        exclude=True,
    )

    scorers: list[ScorerConfig] = Field(
        default=[
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
            ),
            ScorerConfig(
                plugin="heretic.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
        ],
        description=(
            "List of scorer plugin configs. Each entry is an object "
            "{ plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }. "
            '<optimization> is one of "minimize", "maximize", or "none" (do not optimize).'
        ),
    )

    modifiers: list[ModifierConfig] = Field(
        default=[
            ModifierConfig(
                plugin="heretic.modifiers.ara.ARA",
            ),
        ],
        description=(
            "List of modifier plugin configs. Each entry is an object "
            "{ plugin = <plugin>, instance_name = <optional> }. "
            "Note that only a single modifier can currently be applied, "
            "and this list must contain exactly one entry."
        ),
    )

    n_trials: PositiveInt = Field(
        default=100,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: NonNegativeInt = Field(
        default=30,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for reproducible optimization. "
            "Applies to Python's random module, NumPy, PyTorch, and Optuna."
        ),
    )

    study_checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save and load study progress to/from.",
        exclude=True,
    )

    benchmarks: list[BenchmarkSpecification] = Field(
        default=[
            BenchmarkSpecification(
                task="agieval",
                name="AGIEval",
                description="A Human-Centric Benchmark for Evaluating Foundation Models",
            ),
            BenchmarkSpecification(
                task="bbh",
                name="BIG-Bench Hard (BBH)",
                description="Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            ),
            BenchmarkSpecification(
                task="commonsense_qa",
                name="CommonsenseQA",
                description="A Question Answering Challenge Targeting Commonsense Knowledge",
            ),
            BenchmarkSpecification(
                task="eq_bench",
                name="EQ-Bench",
                description="An Emotional Intelligence Benchmark for Large Language Models",
            ),
            BenchmarkSpecification(
                task="gsm8k",
                name="GSM8K",
                description="Training Verifiers to Solve Math Word Problems",
            ),
            BenchmarkSpecification(
                task="hellaswag",
                name="HellaSwag",
                description="Can a Machine Really Finish Your Sentence?",
            ),
            BenchmarkSpecification(
                task="ifeval",
                name="IFEval",
                description="Instruction-Following Evaluation for Large Language Models",
            ),
            BenchmarkSpecification(
                task="mmlu",
                name="MMLU",
                description="Measuring Massive Multitask Language Understanding",
            ),
            BenchmarkSpecification(
                task="mmlu_pro",
                name="MMLU-Pro",
                description="A More Robust and Challenging Multi-Task Language Understanding Benchmark",
            ),
            BenchmarkSpecification(
                task="piqa",
                name="PIQA",
                description="Reasoning about Physical Commonsense in Natural Language",
            ),
            BenchmarkSpecification(
                task="winogrande",
                name="WinoGrande",
                description="An Adversarial Winograd Schema Challenge at Scale",
            ),
        ],
        description="Benchmarks to offer to the user for evaluating abliterated models.",
        exclude=True,
    )

    max_shard_size: PositiveInt | str = Field(
        default="5GB",
        description="Maximum size for individual safetensors files generated when exporting a model.",
    )

    export_strategy: ExportStrategy | None = Field(
        default=None,
        description='How to export the model: "merge", "adapter", or unset to prompt the user.',
    )

    checkpoint_action: str | None = Field(
        default=None,
        description='Action to take in case a checkpoint exists: "continue", "restart", or unset to prompt the user.',
    )

    trial_index: NonNegativeInt | None = Field(
        default=None,
        description="Index (in the sorted Pareto front) of the trial to use, or unset to prompt the user.",
    )

    n_additional_trials: PositiveInt | None = Field(
        default=None,
        description="Number of additional trials to run, or unset to prompt the user.",
    )

    model_action: str | None = Field(
        default=None,
        description='Action to take with the decensored model: "save", "upload", or unset to prompt the user.',
    )

    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_id: str | None = Field(
        default=None,
        description="Name of the Hugging Face repository to upload the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_private: bool | None = Field(
        default=None,
        description="Whether the Hugging Face repository to upload the model to should be private, or unset to prompt the user.",
    )

    upload_reproducibility_information: str | None = Field(
        default=None,
        description='Which reproducibility information to add to the Hugging Face repository: "full", "basic", "none", or unset to prompt the user.',
    )

    ignore_mismatches: bool | None = Field(
        default=None,
        description="Whether to attempt to reproduce the model even if there are environment mismatches, or unset to prompt the user.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    # We intentionally allow extra keys so users can provide plugin-specific
    # configuration in TOML tables like `[scorer.KeywordRate]` which are later
    # consumed via `settings.model_extra` (see `plugin.get_plugin_namespace`).
    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Used during resume - should override *all* other sources.
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="HERETIC_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="config.toml"),
        )
