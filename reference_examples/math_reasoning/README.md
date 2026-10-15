# Math Reasoning Harness Optimization

Meta-Harness experiment for comparing domain-specific and global harness
optimization on IMO-AnswerBench.

Candidate harnesses can retrieve from the external solved-math corpus configured
under `memory` in `config.yaml`. The corpus is loaded lazily, and only problem,
solution, answer, and source fields are exposed. Evaluation ground truth is not
passed to candidates. Task-specific candidates receive the run-level domain;
global candidates and their proposer are domain-blind. Subcategory labels are
hidden in both modes. The Hugging Face corpus revision is pinned in the config
so all runs use the same memory snapshot.

The versioned split is deterministic and balanced across algebra,
combinatorics, geometry, and number theory:

| Split | Per domain | Total |
| --- | ---: | ---: |
| Search | 30 | 120 |
| Development holdout | 20 | 80 |
| Final test | 50 | 200 |

The twelve problems used by the earlier pilot are assigned to search and cannot
appear in development or final test.  Inspect or validate the split without any
API calls:

```bash
cd reference_examples/math_reasoning
uv run pytest -q
```

Run from WSL and use the existing Linux environment from `math-frontier-eval`.
Running `python inner_loop.py` with a Windows Python environment is unsupported
because `llm-provider` uses Linux-only modules. From the MoHarness root in
PowerShell, evaluate the no-retrieval baseline with:

```powershell
./scripts/run_math_search_baseline.ps1 -Domain algebra
```

Run all four domains sequentially under a versioned run name with:

```powershell
./scripts/run_math_search_baselines.ps1
```

The default run name is `baseline-medium-16384-judge-v2`. Version 2 raises the
judge completion budget from 1,024 to 2,048 tokens and makes the equivalence
criteria stricter. Completed domain
result files are skipped when the sequence is restarted.

The equivalent command inside WSL is:

```bash
../../../math-frontier-eval/.venv-linux/bin/python inner_loop.py \
  --candidate agents/no_memory.py \
  --split search \
  --domain algebra \
  --output logs/baseline-medium-16384-judge-v2/algebra/no_memory/search.json
```

Omit `--domain` to evaluate the global, four-domain pool.  `inner_loop.py`
loads generation and symbolic grading from the sibling `math-frontier-eval`
checkout.  This command calls the configured model and therefore requires
`OPENAI_API_KEY`.

Progress is printed after every `experiment.progress_batch_size` problems
(default: 2), including completed problems, elapsed time, and cumulative input
and output tokens. It also reports the cumulative number of empty visible
responses, which can indicate that reasoning consumed the completion budget.
Keep this value at or above the configured concurrency for
efficient parallel execution.

Each successful batch is checkpointed next to the requested output as
`<output>.partial.json`. If a network or API error interrupts a run, execute the
same command again to resume from the last completed batch. The partial file is
removed after the final result is written successfully.

After deterministic grading, a lightweight LLM judge runs only for non-empty
responses marked incorrect. Solver and judge usage are stored separately as
`solver_*_tokens` and `judge_*_tokens`; `total_*_tokens` is their sum. Empty
solver responses are never sent to the judge.

See `domain_spec.md` for the harness boundary, evaluation protocol, and open
budget decisions.  Do not use the final-test split during development.

After estimating the baseline cost, run one global search iteration with:

```powershell
./scripts/run_math_optimizer.ps1 `
  -Mode global `
  -RunName global-smoke-001 `
  -Iterations 1
```

Run a domain-specific iteration with, for example:

```powershell
./scripts/run_math_optimizer.ps1 `
  -Mode task-specific `
  -Domain algebra `
  -RunName task-specific-algebra-001 `
  -Iterations 1
```

One iteration evaluates three generated candidates.  The optimizer only uses
the search split; development selection and final-test evaluation remain
explicit later phases to prevent accidental leakage.

Use `--candidates-per-iteration 1` for a low-cost end-to-end smoke test. Each
run stores generated candidates below its own `logs/<run-name>/agents/`
directory. If execution stops after a completed candidate or while a manifest
is pending, repeat the same command with `--resume`; completed candidates are
not evaluated again. Frozen no-memory search baselines are configured under
`baseline.search_results` in `config.yaml`; they are not selected through a
command-line argument. Task-specific runs use their domain file, while global
runs combine the same four domain files without new solver calls. Source paths,
SHA-256 hashes, and compatibility status are copied into the run logs.

Every new optimization run writes `run_manifest.json` with the full config,
baseline hashes, repository revisions and dirty-state fingerprints, and safe
authentication metadata. Each resume appends to `run_invocations.jsonl`.
Baseline files produced before per-result `evaluation_config` was introduced
are accepted only because `baseline.allow_legacy_without_evaluation_config` is
explicitly enabled; their compatibility status is `legacy_partial` rather than
`verified`.

To migrate an earlier baseline to new grading while regenerating only known
invalid solver responses, pass `-ReuseResult` and one or more
`-RegenerateId` values to `run_math_search_baseline.ps1`. Existing partial
responses in the destination run take precedence over reused responses.

## Development selection and final test

The optimizer evaluates the no-memory baseline and every generated candidate
on both search and development splits. Search problem-level traces are exposed
to the proposer for diagnosis. Development problem text, responses, answers,
IDs, and per-problem correctness remain hidden; only candidate-level aggregate
accuracy, token use, and empty-response counts enter the next iteration.

After every candidate, the optimizer updates `logs/<run-name>/frontier_val.json`
using development accuracy and cost. To backfill or rebuild development results
for a run created by an older optimizer, use:

```powershell
./scripts/run_math_selection.ps1 `
  -Phase dev `
  -Mode task-specific `
  -Domain combinatorics `
  -RunName task-specific-combinatorics-smoke-002
```

The primary selection metric is accuracy for task-specific runs and four-domain
macro accuracy for global runs. The Pareto cost axis is solver input plus output
tokens per problem; judge tokens are evaluation overhead and are not included.
Candidate-level dev files are resumable. Development aggregates may guide later
iterations, so dev is an adaptive validation set rather than a final unbiased
holdout.

Before opening final test, exercise the complete finalization path on two
already-exposed search problems:

```powershell
./scripts/run_math_selection.ps1 `
  -Phase smoke `
  -Mode task-specific `
  -Domain combinatorics `
  -RunName task-specific-combinatorics-smoke-002 `
  -SmokeProblems 2
```

This runs the baseline, best-dev system, and Pareto systems through candidate
loading, retrieval, generation, grading, judging, checkpointing, and resume.
Outputs go under `smoke_finalization/`; no test rows are loaded and
`finalized.json` is not created.

Only after selection is final, explicitly evaluate the no-memory baseline,
best-accuracy candidate, and all Pareto candidates on final test:

```powershell
./scripts/run_math_selection.ps1 `
  -Phase test `
  -Mode task-specific `
  -Domain combinatorics `
  -RunName task-specific-combinatorics-smoke-002
```

The test phase requires `frontier_val.json`, writes `finalized.json` before its
first model call, and resumes candidate-by-candidate after interruption. Test
results never feed back into the proposer or candidate selection.

