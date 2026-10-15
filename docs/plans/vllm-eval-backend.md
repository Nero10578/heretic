# Plan: vLLM inference backend for Heretic (ARA workflow) evaluations

## 1. Goal

Route the per-trial **scorer evaluations** (response generation + capture-position
log-probabilities) through a **vLLM** engine, while keeping the **modifier stage**
(ARA `_modify_model`, one-time module-I/O capture) and **export** on HuggingFace.

The evaluations dominate trial wall-clock (~3h30m/trial, ~100 prompts x long reasoning
generations, batch 8 on HF `generate`). They are pure inference, so paged attention,
continuous batching, CUDA graphs and optional TP apply directly.

### Non-goals
- Moving ARA's optimization (`ara.py:336-447`) or module-I/O capture to vLLM: both need
  full weights, autograd, and writes, which vLLM does not expose.
- Implementing tensor parallelism inside Heretic: vLLM owns TP.
- Changing the optimizer/objective or the final exported artifact.

## 2. Verified facts this plan relies on

- **The ARA artifact is a normal PEFT LoRA.** `apply_lora` builds
  `LoraConfig(r=lora_rank, lora_alpha=lora_rank)` (`model.py:624-633`), so PEFT scaling
  = 1; the model is never weight-modified; export is `model.model.save_pretrained(...)`
  (`main.py:949-954`). vLLM computes `scaling = lora_alpha / rank`
  (`vllm/lora/lora_weights.py:33`) = 1 here, so `LoRARequest` reproduces heretic's
  evaluated transformation exactly.
- **LoRA on a GPTQ base is supported.** `AutoGPTQLinearMethod` (`auto_gptq.py:301`) is
  reached through the LoRA wrapper's `layer._get_quant_method().apply(...)`
  (`vllm/lora/layers/column_parallel_linear.py:36`), and `_get_lora_device` explicitly
  handles quantized base layers (`vllm/lora/layers/utils.py:45-51`). No "LoRA not
  supported with GPTQ" guard exists.
- **Full-vocab logprobs are supported, and raw logits too.** Engine `max_logprobs = -1`
  means "no cap, i.e. all (output_length * vocab_size) logprobs are allowed to be
  returned" (`vllm/config/model.py:282-285`); `SamplingParams(logprobs=-1)` requests the
  full vocab (`vllm/sampling_params.py:919-934`). `logprobs_mode` selects the returned
  quantity: `raw_logprobs` (default, pre-temperature log-softmax), `processed_logprobs`,
  `raw_logits`, `processed_logits` (`vllm/config/model.py:286-291`). No vLLM change
  needed. Cost caveat, per the same docstring: full-vocab output at *every* position
  "may cause OOM", which is why only one position is requested (section 4.1).
- **Per-request adapter hot-swap** via `LoRARequest` (`vllm/lora/request.py:8`).
- **Outputs carry `token_ids` and `logprobs`** per generated position
  (`vllm/outputs.py:57,81`), enough to locate capture positions and read distributions.
- **Gemma-4 + LoRA is supported** (`vllm/model_executor/models/config.py:1065`,
  `adapters.py:183`).
- **No bitsandbytes** path exists in this build (no `load_in_4bit`, no
  `quantization/bitsandbytes.py`) — the HF `bnb_4bit` model cannot be handed to vLLM;
  use the pre-quantized GPTQ base.
- **The Context is the only seam scorers use.** Scorers call `ctx.get_generation_records`,
  `ctx.get_responses`, `ctx.get_logits` (`plugin.py:192-273`); all delegate to
  `self._model`.
- **Evaluator builds contexts** for init/baseline (`evaluator.py:44`) and per trial
  (`evaluator.py:150`); the modifier builds its own for init (`modifier.py:226`) and
  `main.py` builds one per trial (`main.py:617`, `870`).
- **vLLM is installed** in the heretic venv (dev build), though the local Windows venv's
  torch is CPU-only, so verification runs happen on GPU servers.

**vLLM does not need to be modified.** Every capability this plan needs is already
exposed by the public API of the installed checkout (GPTQ base + LoRA, per-request
`LoRARequest`, full-vocab logprobs via `max_logprobs = -1`, `token_ids`/`logprobs` in
`CompletionOutput`, in-process `LLM` so Heretic can pass pre-rendered prompt strings).
All new code lives in Heretic.

## 3. Architecture

Introduce a small **inference-backend seam at `Context`**; do not touch the `Model`
class's HF logic.

### 3.1 Backend protocol

New module `src/heretic/inference.py`:

```python
class InferenceBackend(Protocol):
    def get_generation_records(
        self, prompts: list[Prompt], *, capture: CaptureRequest
    ) -> list[GenerationRecord]: ...
    def get_logprobs(
        self, prompts: list[Prompt], after_reasoning: bool | None
    ) -> Tensor: ...  # full-vocab log-probs at the capture position
```

- `HFBackend`: thin wrapper over `Model.get_response_records_batched` and
  `F.log_softmax(Model.get_logits_batched(...))`. This is the current behavior and the
  default.
- `VLLMBackend` (new `src/heretic/vllm_backend.py`): described below.

`Context.__init__(settings, model, backend=None)`:
- `backend` defaults to `VLLMBackend` when `settings.vllm.enabled`, else `HFBackend`.
- `get_generation_records` / `get_responses` route to `self._backend`.
- `get_logprobs` (new) routes to `self._backend`.
- `get_logits`, `get_residuals*`, `get_module_io` **always** use `self._model` (HF),
  because vLLM cannot produce them. `get_logits` remains for the non-reasoning path and
  any consumer that wants raw logits.

### 3.2 Which contexts get the vLLM backend

- Scorer contexts (`evaluator.py:44` for init/baseline, `evaluator.py:150` for trials):
  **vLLM** (text + logprobs only).
- Modifier contexts (`modifier.py:226`, `main.py:617`, `main.py:870`): **HF**, because ARA
  relies on `capture_module_io_default = True` and `get_module_io`
  (`ara.py:228-259`). These must never be given the vLLM backend.

This split is the key invariant: **module-I/O-bearing generations stay on HF; pure eval
generations go to vLLM.**

### 3.3 Adapter sync (per trial)

After `modify_model` writes A,B into the HF PEFT modules, the backend must present the
same adapter to vLLM:

1. Extract the adapter from `model.model` (PeftModel) into an adapter artifact
   (`adapter_config.json` + `adapter_model.*`).
2. **Normalize `target_modules`** from heretic's fully-qualified names
   (`model.py:608-622` collects `full_name`) to suffixes (`o_proj`, `down_proj`) that
   vLLM binds (`docs/features/lora.md:480`). Verify vLLM's name mapping for Gemma-4
   (`adapters.py:183`).
3. Confirm `lora_alpha = r` (already true) so vLLM scaling is 1.
4. Register with vLLM via `LoRARequest`; hot-swap per trial (`add_lora`/`remove_lora`,
   engine started with `enable_lora=True`, `max_lora_rank >= rank`,
   `--lora-target-modules`). Adapters are tens of MB, so cost is small; a later
   optimization can use `update_weights` to avoid disk I/O.

### 3.4 Engine lifecycle

- `Evaluator` builds one backend instance (engine load is expensive) and passes it into
  every Context it creates; shut it down at process exit.
- `mode = "inprocess"`: `vllm.LLM(...)` in the Heretic process. **Preferred**, because it
  lets Heretic pass its own tokenizer-rendered prompt strings (with
  `strftime_now=strftime_fixed_date`, `model.py:919`) instead of vLLM's server-side chat
  templating, keeping prompt construction identical to HF.
- `mode = "server"`: point at a running `vllm serve` OpenAI-compatible endpoint
  (`url`), so the optimizer host and inference host can be separate servers. Only the
  adapter crosses the wire.

### 3.5 Init-phase filtering and baselines

The init phase generates too, and it splits exactly along the 3.2 boundary:

- **Scorer init** (`evaluator.py:44-51`): prompt filtering for `KLDivergence`
  (`filter_prompts`, `kl_divergence.py:110-115`) and `KeywordRate`
  (`keyword_rate.py:96-101`), the KL baseline logprobs (`kl_divergence.py:118`), and the
  baseline scores (`get_baseline_score` -> `get_score`) all run in the scorer Context, so
  they go to **vLLM**. This is the base-model generation pass and benefits the most.
  Filtering *must* run on the same engine as scoring, because it selects the prompt subset
  that is then scored; that holds automatically since both share one Context/backend.
- **Modifier init** (`modifier.py:226-229`): `ARA.init`'s optional
  `filter_prompts_by_refusal` (`ara.py:245-248`) stays on **HF**, because it is the same
  generation that fills the module-I/O records for `get_module_io` (`ara.py:255-259`):
  `get_responses` populates the records, then `get_module_io` reuses them, and module I/O
  cannot come from vLLM. It is one-time and cached (`init_cache`), and is only active when
  `ara.filter_prompts = true` (default false).
- **Base model, no adapter**: at init the model is unmodified (`apply_lora` runs inside
  `ARA.init` at `ara.py:221,273`, after the Evaluator at `main.py:573-596`). So init
  generations on vLLM use **no adapter** (or an identity adapter). Baseline logprobs are
  thus the GPTQ base's distribution — the same base the per-trial evals add an adapter onto.
- **Cache correctness**: cached filter selections and baselines are only valid for the
  engine and base that produced them, so `init_cache` keys must include the backend
  identity (section 5). An HF-produced prompt subset must never be reused for a vLLM run.
- **Minor caveat**: if `ara.filter_prompts = true` and the vLLM base differs from the HF
  base (GPTQ vs bf16; see section 6), ARA's prompt selection uses HF-base refusals while
  scoring uses GPTQ-base behavior. Make the two bases identical (option (a) or (b) in
  section 6) if you use ARA filtering.

## 4. Generation parity (must match the HF path)

Replicate heretic's generation exactly:

| heretic | vLLM equivalent |
|---|---|
| `apply_chat_template(add_generation_prompt=True, tokenize=False, strftime_now=strftime_fixed_date, **chat_template_kwargs)` + `response_prefix` (`model.py:901-935`) | Reuse `Model._build_chat_prompts` (shared tokenizer); send rendered strings as text prompts |
| `do_sample=False` greedy (`model.py:956`) | `SamplingParams(temperature=0.0)` |
| `max_new_tokens = max_thought_length + max_response_length` (`model.py:1023-1030`) | `SamplingParams(max_tokens=...)` |
| EOS/pad semantics | `SamplingParams` EOS handling + same stop tokens |
| Repetition abort (`model.py:129-190`) | **Intentionally dropped** (decision: vLLM is fast enough that a runaway generation is acceptable). Do not port the processor. |
| capture ids via `_answer_and_capture_ids` / `_first_content_ids` (`model.py:1306-1397`) | Compute from `output.token_ids` using the same Model helpers |

### 4.1 Capture-position -> logprob mapping

`capture_ids = prompt_ids + generated_ids[:answer_start]` (`model.py:1391-1395`). The
distribution heretic captures at that position is the model's next-token distribution.

**Do not request `logprobs = vocab_size` for the whole generation**: that returns a full
~2e5-vocab distribution at every generated token (thousands of tokens x 100 prompts),
which is prohibitive to transfer and hold. Instead:

1. Main generation: request cheap top-1 logprobs (optionally) only to decode and locate
   the capture positions from `token_ids`.
2. Second, generation-free request per prompt: feed `capture_ids` as the prompt with
   `max_tokens = 1` and `logprobs = vocab_size`. The single output position's logprobs
   are exactly the full capture distribution. Cost is one prefill per prompt, which vLLM
   batches and pipelines cheaply.

This is the vLLM analogue of heretic's HF fallback `get_logits_from_capture_ids`
(`model.py:1790`), and it needs no vLLM change. Rows whose reasoning never closed fall
back to the HF capture path (`Context._assemble_logits`, `plugin.py:275-311`).

## 5. KL scorer refactor (engine-agnostic)

`KLDivergence` currently does `F.log_softmax(ctx.get_logits(...))` twice and stores
`_baseline_logprobs` (`kl_divergence.py:118-139`). Raw logits are not directly available
from vLLM, but log-probs are. Refactor to a **log-prob contract**:

- Add `Context.get_logprobs(prompts, after_reasoning) -> Tensor`.
  - HF: `F.log_softmax(get_logits(...))`.
  - vLLM: assemble the full-vocab log-probs from step 4.1; HF fallback when missing.
- Change `KLDivergence.init/get_score` to use `get_logprobs` directly, dropping the
  internal `log_softmax`. Baseline semantics are unchanged (already log-probs).
- Cache key (`cache.py: environment_key`) must include the backend identity
  (HF vs vLLM model/quant/url) so an HF baseline is never reused for a vLLM run.

This removes the requirement for raw logits from vLLM and makes the exact same scorer
work on both engines. **No vLLM source changes are required.**

## 6. Configuration

Add `VLLMSettings(BaseModel)` to `config.py` and a `vllm: VLLMSettings` field on
`Settings` (documented in `config.default.toml` / `config.toml.example`):

```toml
[vllm]
enabled = false
model = ""                      # pre-quantized base for vLLM; default: use `model`
mode = "inprocess"              # "inprocess" | "server"
url = ""                        # server mode (OpenAI-compatible)
dtype = "auto"                  # e.g. "gptq" for the pre-quantized Gemma weights
tensor_parallel_size = 1
gpu_memory_utilization = 0.9
max_model_len = 0               # 0 = auto
max_lora_rank = 64              # >= max modifier lora_rank
lora_target_modules = ["o_proj", "down_proj"]
max_logprobs = -1               # engine-side; enables full-vocab requests
logprobs_mode = "raw_logprobs"  # raw_logprobs | raw_logits | processed_*
enforce_eager = false
trust_remote_code = false
```

`model`/`dtype` point at the pre-quantized GPTQ Gemma base; vLLM loads it and applies the
ARA adapter as a LoRA on top, with no vLLM change.

**Base parity (accepted tradeoff).** The ARA optimizer runs on the *HF* base
(`W_base` at `ara.py:354`), while vLLM evaluates on its own base (e.g. GPTQ). Using
different bases is acceptable: ranking stays coherent because every candidate is scored on
the same vLLM base, and the artifact is the adapter, which applies to either base. If you
want the two to share a base, either point vLLM at the bf16 base (a deployment choice) or
have the modifier dequantize the GPTQ base (mirroring the bnb `quant_state` path at
`ara.py:355-364`).

The one case that argues for matching bases is `ara.filter_prompts = true` (section 3.5):
the filter labels prompts by HF-base refusals while scoring measures vLLM-base refusals.
`bnb_4bit` also cannot be handed to vLLM in this build, so an NF4 HF base with
`vllm.enabled` requires pointing vLLM at a loadable base (bf16 or GPTQ).

## 7. Files

**New**
- `src/heretic/inference.py` — `InferenceBackend` protocol + `HFBackend`.
- `src/heretic/vllm_backend.py` — `VLLMBackend`: engine lifecycle, adapter sync +
  `target_modules` normalization, prompt building, generation, capture positioning,
  full-vocab logprob assembly.
- `tests/test_vllm_backend.py` — GPU-free unit tests with a stubbed `vllm` module.

**Modified**
- `src/heretic/plugin.py` — `Context` backend seam; new `get_logprobs`.
- `src/heretic/evaluator.py` — build/share the backend across scorer contexts.
- `src/heretic/scorers/kl_divergence.py` — use `get_logprobs`.
- `src/heretic/model.py` — expose id helpers (`_build_chat_prompts`,
  `_answer_and_capture_ids`, `_first_content_ids`, `_reasoning_close_tag_ids`,
  `_eos_token_ids`, `decode_tokens`, `_response_max_new_tokens`) for reuse; no behavior
  change.
- `src/heretic/config.py`, `config.default.toml`, `config.toml.example` — vLLM settings.
- `src/heretic/cache.py` — backend in `environment_key`.
- `src/heretic/main.py` — construct backend, pass to `Evaluator`, shut down.
- `pyproject.toml` — optional `[project.optional-dependencies] vllm = ["vllm"]`.

## 8. Phases

**Phase 0 — Seam, no behavior change.**
Add `InferenceBackend` + `HFBackend`; route `Context` through it; default HF everywhere.
All 59 existing tests stay green; `ruff format`/`ruff check --extend-select I`/`ty check`
clean (mirror `.github/workflows/ci.yml`).

**Phase 1 — KeywordRate through vLLM (text only).**
`VLLMBackend.get_generation_records` returning `response` + `capture_ids` (no logprobs).
Adapter sync on the GPTQ base, engine lifecycle, config plumbing. A/B on a fixed prompt
set: same adapter, compare refusal rate HF vs vLLM (greedy should be near-identical).

**Phase 2 — KL through vLLM.**
`Context.get_logprobs` + KL refactor. Full-vocab logprobs via the second 1-token request,
capture positioning, HF fallback. Validate `KL_vllm ~= KL_hf` within tolerance; baseline
recomputed in vLLM.

**Phase 3 — Scale & robustness.**
`tensor_parallel_size > 1`; server mode; cache key with backend; optional `update_weights`
to skip adapter disk I/O.

**Phase 4 — Docs & CI.**
Docs page + `config.toml.example`. CI has no GPU: tests must stub `vllm` and guard the
import so dependency/lint/test jobs are unaffected.

## 9. Risks / open questions (verify while implementing)

1. **Logprob semantics — resolved.** `logprobs_mode = "raw_logprobs"` (default) returns
   the raw, pre-temperature log-softmax; `"raw_logits"` returns raw logits
   (`vllm/config/model.py:286-291`). Set it explicitly so KL parity does not depend on the
   default; `raw_logits` matches heretic's `get_logits` most closely.
2. **Full-vocab delivery**: confirm `logprobs = vocab_size` with `max_logprobs = -1`
   returns all entries for the single capture position.
3. **Adapter hot-swap at scale**: `add_lora`/`remove_lora` behavior over hundreds of
   trials (memory, reuse of a fixed slot under `max_loras`).
4. **Target-module binding**: heretic's fully-qualified `target_modules` vs vLLM's
   suffix/regex matching for Gemma-4.
5. **Greedy parity on a GPTQ base**: token-for-token divergence between HF (bf16/NF4) and
   vLLM (GPTQ); expected to be a systematic offset, not noise. See the base-parity caveat
   in section 6.
6. **Coexistence**: HF model (ARA) + vLLM engine resident together — fine on the target
   servers; ARA needs the HF model regardless.

## 10. Testing

- **Unit (no GPU)**: prompt-construction parity; id -> capture-position mapping;
  adapter `target_modules` normalization; KL scorer on the log-prob contract; cache key
  includes backend. Stub `vllm` so CI stays green.
- **Integration (GPU)**: parity script over ~20 prompts comparing HF vs vLLM for refusal
  rate and KL; determinism check across repeated runs.
- **CI**: `ruff format --check .`, `ruff check --extend-select I .`,
  `ty check --error-on-warning .`, unittest discover.

## 11. First milestone (smallest useful increment)

Phase 0 + Phase 1 for the **KeywordRate** scorer only, on one GPU, with a fixed ARA
adapter exported once against the GPTQ base: prove refusal rates match HF, then proceed
to the KL logprob path. This de-risks adapter sync and generation parity before touching
logprobs.

## 12. Does vLLM need to be modified?

No. Required capabilities and their existing entry points:

| Need | Provided by |
|---|---|
| Load pre-quantized Gemma-4 GPTQ base | `AutoGPTQConfig` / `Gemma4Config` |
| Apply ARA adapter on the GPTQ base | LoRA wrapper `_get_quant_method().apply(...)` |
| Hot-swap adapter per trial | `LoRARequest` + `add_lora`/`remove_lora` |
| Full-vocab distribution (or raw logits) at a position | `max_logprobs = -1`, `logprobs = -1`, `logprobs_mode` on a 1-token request |
| Generated token ids + logprobs | `CompletionOutput.token_ids` / `.logprobs` |
| Pre-rendered prompt strings (template parity) | in-process `LLM` with `TextPrompt` |
| Greedy determinism | `SamplingParams(temperature=0, seed=...)` |

Everything new is Heretic-side; the vLLM checkout only needs to be installed and on the
GPU hosts.

## 13. Implementation status

- **Phase 0 — done.** `InferenceBackend` + `HFBackend` (`src/heretic/inference.py`);
  `Context` routes generation through an injected backend, defaulting to HF with no
  behavior change (`plugin.py:174`).
- **Phase 1 — done.** `VLLMBackend` (`src/heretic/vllm_backend.py`): lazy in-process
  engine, per-trial adapter export with `target_modules` suffix normalization and a
  `LoRARequest`, greedy generation returning `GenerationRecord`s with capture ids.
  `VLLMSettings` / `[vllm]` added (`config.py`, `config.default.toml`,
  `config.toml.example`); `Scorer.text_only`; the `Evaluator` builds and shares the
  backend and syncs the adapter each trial (`evaluator.py`); the init-cache key includes
  the backend identity (`cache.py`). Tests: `tests/test_vllm_backend.py` (GPU-free).
  Repo green: 70 tests, `ruff` and `ty` clean.
- **Phase 2 — done.** `Context.get_logprobs` (`plugin.py`) is the log-prob contract KL now
  uses; `KLDivergence` calls it instead of `get_logits` + `log_softmax`
  (`kl_divergence.py`). `VLLMBackend.get_logprobs_prompt` /
  `get_logprobs_from_capture_ids` fetch one full-vocab distribution per capture position
  via a generation-free 1-token request (`logprobs=-1`), honoring `logprobs_mode`
  (`raw_logprobs` directly, or `raw_logits` converted). `Scorer.text_only` was replaced by
  `requires_model_internals` (default True); `KeywordRate` and `KLDivergence` decline it,
  so the backend is now selected for the config in `config.toml.example`. Verified: 74
  tests, `ruff` and `ty` clean.
- **Selector.** vLLM is used when enabled and no configured scorer requires model internals
  (`BenchmarkScore` does, and stays on HuggingFace).
- **Resume with a backend swap.** On "continue", the `[vllm]` settings are taken from the
  current configuration (overlaid on the saved run) rather than the checkpoint, with a
  notice when they differ. Modifier init caches no longer include the backend identity
  (`Plugin.backend_dependent_init = False`; `Scorer` overrides it to True), so changing
  the vLLM model reuses ARA's module-I/O capture and only recomputes the scorer baselines.
  Trials scored before and after the swap are not directly comparable.
- **Mismatched-cache reuse.** On a cache miss, `InitCache.load_with_fallback` discovers
  entries for the same plugin under other environment keys, prints the settings that
  differ, and (interactively, or via an injected `decide`) reuses one if asked. Guards
  against a code edit or a settings change silently costing hours of re-initialization.
  Environment payloads are now recorded as `_environment.json` sidecars so a mismatch can
  be explained (`environment_payload`/`environment_key` split in `cache.py`).
- **Sleep mode for tight VRAM.** `[vllm] sleep_for_modifier` enables vLLM's sleep mode
  (`enable_sleep_mode` in `engine_kwargs`) and wraps each per-trial modification with
  `pause()`/`resume()` (`sleep(level=1)` / `wake_up`) in the trial loop and the export
  path, inside `try`/`finally` so the engine is always woken. The scoring engine releases
  the GPUs for the modifier and returns for scoring, so the two no longer have to fit
  together. `InferenceBackend.pause`/`resume` are no-ops on the HuggingFace backend.
- **Outstanding.** Phase 4 (docs), tensor parallel / server mode, adapter lifecycle at
  scale, and GPU validation of the engine, adapter hot-swap, editor parking, and sleep
  mode (risks 3 and 4 in section 9).

## 14. Editor parking (single-GPU serialization)

Goal: let a full-precision HuggingFace editor and a quantized vLLM tester share one GPU,
without ever reloading either engine.

- **Config**: `[vllm] park_editor = false`. Active when true and vLLM is enabled.
- **Startup order**: the modifier is initialized first (its one-time ARA module-I/O capture
  needs the GPU), then `Model.park()` moves every editor parameter and buffer to CPU
  memory, and only then does the evaluator's generating init run and create the vLLM
  engine on the now-free GPU. Implemented via `Evaluator(defer_init=...)` /
  `Evaluator.initialize()` and the gated branch in `main.py`.
- **Per-trial**: the editor stays parked; ARA's `_modify_model` stages each module's LoRA
  parameters, base weight, and cached prompt I/O onto an accelerator for the optimization
  and moves the optimized parameters back to their storage device afterwards
  (`ARA._compute_device`). The default GPU-resident path is unchanged because staging onto
  the already-current device is a no-op.
- **vLLM**: loads once and owns the GPU for the rest of the run; never reloaded per trial.

Limitations, to validate on a GPU:
- Only unquantized HuggingFace editors: the bitsandbytes dequantization path is not staged
  onto the compute device, so a parked 4-bit editor is unsupported (the bf16 editor is the
  intended case).
- `Model.park()` moves parameters individually and does not reconcile Accelerate's dispatch
  hooks; the editor must not run a forward while parked (nothing does in this flow).
- Requires host RAM to hold the editor (~62 GB for a 31B bf16 model).
- GPU validation of the staging and the startup reorder is still outstanding.
