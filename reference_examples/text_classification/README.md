# Text Classification

Minimal text classification reference experiment for Meta-Harness. The outer loop writes candidate memory systems to `agents/`; the inner loop evaluates them on the datasets in `config.yaml`.

## Quick Start

Install:

```bash
cd reference_examples/text_classification
uv sync
```

Run one evolve iteration:

```bash
uv run python meta_harness.py --iterations 1
```

Run one memory system on one dataset:

```bash
RUN=logs/manual
OUT="$RUN/Symptom2Disease/fewshot_all/gpt-6-luna"
mkdir -p "$OUT"
PYTHONPATH=.. uv run python -m text_classification.inner_loop \
  --memory fewshot_all \
  --dataset Symptom2Disease \
  --model openai/gpt-6-luna \
  --val-output "$OUT/val.json" \
  --save-memory "$OUT/memory.json"
uv run python benchmark.py --logs-dir "$RUN" --results
```

This pilot uses `openai/gpt-6-luna` from `config.yaml` and requires
`OPENAI_API_KEY`. The outer-loop proposer uses the authenticated Codex CLI;
run `codex login` before starting an evolution run. To target another provider
or any OpenAI-compatible endpoint, override `--model` and optionally
`--api-base`.

Print the benchmark summary:

```bash
uv run python benchmark.py --results
```

Plot accuracy against memory-context size and highlight the Pareto frontier:

```bash
uv run python plot_pareto.py --run-name specific-uspto-002
```

The SVG is written to `logs/<run-name>/pareto_plot.svg`. For a multi-dataset
run, the plot uses the unweighted macro-average across datasets. Use repeated
`--dataset` arguments to select a subset, and `--x-scale linear` if a linear
x-axis is preferable to the default log scale.

Evolution uses validation results only. Finalize a named run explicitly after
selection is complete:

```bash
uv run python meta_harness.py --run-name my-run --test
```

Test results are written under `logs/my-run/results/` and printed at the end.
Finalization permanently blocks more evolution under that run name. The test
data is included in this public repository, so this is operational isolation,
not access control.

Run provider-free tests:

```bash
PYTHONPATH=.. uv run python -m unittest discover -s tests -v
```

## Layout Notes

The `proposer` section in `config.yaml` selects the outer-loop Codex model and
reasoning effort (pilot defaults: `gpt-6-sol`, `high`). The `models` section
selects the inner-loop solver separately. Each proposer invocation records its
requested `model` and `reasoning_effort`, token usage, duration and exit code in
`logs/<run-name>/codex_sessions/<timestamp>_iter<N>/meta.json`.
These fields record the requested settings, not a resolved model snapshot.

Set `proposer.dataset_blind: true` to optimize one shared harness using only the
unweighted macro-average validation score. In this mode the proposer runs in an
isolated view containing reusable agent source plus `optimizer_state.json`; it
does not receive dataset names, task boundaries, or per-dataset scores. It does
receive shuffled problem-level traces in `anonymous_traces.jsonl`, so it can
study inputs, prompts, predictions, labels, and errors and may infer task
structure from their contents. Set it to `false` to retain the original
dataset-aware workflow. Use a new run name when changing this setting.

Each entry in `models` can specify `reasoning_effort` (pilot: `medium`).
LiteLLM is pinned to 1.101.0 for GPT-6 request support. For GPT-6 with reasoning
enabled (including an unspecified effort), temperature is omitted from requests.
To use temperature 0, explicitly set `reasoning_effort: none` for Sol/Luna.
The evaluator reads the entry matching the selected model and forwards the value
to LiteLLM. A direct `inner_loop` invocation can override it with
`--reasoning-effort`. The value is recorded in evaluation JSON and training-log
metadata; it also participates in the response cache key. Omitting it leaves the
provider default unchanged. Use a new run name when changing model settings,
because completed benchmark result files are reused within a run.

- `agents/`: the kept baselines plus the write target for generated candidates.
- `.claude/skills/meta-harness/SKILL.md`: main proposer prior used by `meta_harness.py`.

## Runtime And Cost

This pilot targets OpenAI (`openai/gpt-6-luna`). If you want a different
provider or your own OpenAI-compatible endpoint, pass `--model` and optionally
`--api-base`, or change `config.yaml`. The paper experiments used a local
`vllm` deployment of `gpt-oss-120b`, MXFP4 quantized, with
`max-model-len=32768`. API-backed runs may differ in quality from that setup.

## Release Notes

- `config.yaml` is the source of truth for datasets, models, and active memory systems.
- The public release includes the MCE paper datasets for this experiment under `data/`, so there is no runtime clone step.
- `inner_loop.py` still uses package-mode imports, so the single-candidate command above keeps `PYTHONPATH=..` when run from this directory.
- `benchmark.py` is the sweep/orchestration layer used by `meta_harness.py`; `inner_loop.py` is the single memory-system evaluator that `benchmark.py` dispatches.
