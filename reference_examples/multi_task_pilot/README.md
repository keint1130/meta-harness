# Multi-task Pilot

Pilot environment for comparing one global harness with task-specific harnesses
on coding, retrieval, and mathematical reasoning.

## Fixed task families

- Coding: LiveCodeBench `code_generation_lite`, fixed `v6` release window,
  Python answers. This is the newest-problem window (`test6.jsonl`), not the
  cumulative `release_v6` dataset.
- Retrieval: the PopQA-derived portion of RetrievalQA (1,570 rows in the pinned
  revision), with frozen pre-retrieved Contriever documents. The original
  RetrievalQA paper used an earlier filtered set of 659 PopQA questions. The
  original PopQA repository's published
  BM25 Google Drive link currently returns 404, so it is not a reproducible
  dependency for this pilot.
- Math: the Level 5 subset of `HuggingFaceH4/MATH-500`. An initial uniformly
  sampled dev baseline scored 29/30 after correcting evaluator false negatives,
  so the pilot uses the hardest official level to avoid saturation.

The upstream datasets primarily expose evaluation splits. This pilot creates a
deterministic local dev/test split using SHA-256 over `seed:id`, then records the
selected IDs and upstream revisions in a manifest. Scores are pilot-subset
scores, not official full-benchmark scores.

## Isolation

Generated code must only run through `DockerSandbox`. The container has no
network, a read-only root filesystem, dropped Linux capabilities, CPU, memory,
PID, and wall-time limits. Do not add a host-execution fallback.

## Initial experiment order

1. Prepare and freeze dataset manifests.
2. Verify each evaluator with fixtures and one-item smoke tests.
3. Evaluate the shared baseline harness on 30 dev items per family.
4. Optimize task-specific and global harnesses with the same proposal budget.
5. Finalize once on 50 held-out items per family.

Run provider-free tests from this directory:

```powershell
uv run python -m unittest discover -s tests -v
```

## Commands

All commands below are run from
`reference_examples/multi_task_pilot`. Paths use `/` so they work in
PowerShell as well as other shells.

Install the pinned environment and prepare the deterministic subsets:

```powershell
uv sync
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_DATASETS_CACHE = "C:/tmp/moh-hf-datasets"
uv run python prepare_data.py
```

The selected IDs and exact upstream commit hashes are committed under
`manifests/`. The full prepared examples are local and ignored by Git.

Run a low-cost API smoke test (one item from each family):

```powershell
uv run --env-file ../../.env python run_baseline.py --limit 1 --run-name baseline-smoke
```

Then run the complete 30-item-per-family development baseline:

```powershell
uv run --env-file ../../.env python run_baseline.py --run-name baseline-dev
```

Each run writes `logs/<run-name>/results.jsonl` and `summary.json`. The JSONL
contains the answer, score, token usage, tool trace, and any runtime error for
every problem. It also snapshots the exact `config.yaml` and manifests used by
that run. Do not run the held-out split while developing or optimizing a
harness. Final evaluation uses `--split test` once, after the harnesses and
proposal budget have been frozen.

Run one equal-budget proposal for each task-specific track and the global
track after the baselines exist:

```powershell
uv run --env-file ../../.env python optimize.py `
  --baseline-runs baseline-dev baseline-math-level5 `
  --iterations 1
```

The optimizer writes candidates, sanitized failure reports, Codex session
logs, frontier specs, and `evolution.jsonl` under `evolution/<run-name>/`.
Only a strictly higher development score replaces a frontier; ties are
rejected. Candidate specs cannot change the model, reasoning effort, datasets,
budgets, evaluator, or sandbox.

## Experimental controls

- `config.yaml` is the single source of truth for the model, reasoning effort,
  tool/LLM budgets, dataset revisions, split sizes, and Docker limits.
- A candidate harness receives `Task`, never `EvalCase`; reference answers and
  private tests stay evaluator-owned.
- PopQA retrieval can only inspect the frozen contexts attached to that item.
- Candidate Python and LiveCodeBench solutions run only in the networkless
  Docker sandbox.
- Family scores are averaged separately, then combined as an equal-weight
  macro average. This prevents one family from dominating by sample count.
