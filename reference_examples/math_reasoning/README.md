# Math Reasoning Harness Optimization

Meta-Harness experiment for comparing domain-specific and global harness
optimization on IMO-AnswerBench.

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

The equivalent command inside WSL is:

```bash
../../../math-frontier-eval/.venv-linux/bin/python inner_loop.py \
  --candidate agents/no_memory.py \
  --split search \
  --domain algebra \
  --output logs/manual/algebra/no_memory/search.json
```

Omit `--domain` to evaluate the global, four-domain pool.  `inner_loop.py`
loads generation and symbolic grading from the sibling `math-frontier-eval`
checkout.  This command calls the configured model and therefore requires
`OPENAI_API_KEY`.

Progress is printed after every `experiment.progress_batch_size` problems
(default: 2), including completed problems, elapsed time, and cumulative input
and output tokens. Keep this value at or above the configured concurrency for
efficient parallel execution.

Each successful batch is checkpointed next to the requested output as
`<output>.partial.json`. If a network or API error interrupts a run, execute the
same command again to resume from the last completed batch. The partial file is
removed after the final result is written successfully.

See `domain_spec.md` for the harness boundary, evaluation protocol, and open
budget decisions.  Do not use the final-test split during development.

After estimating the baseline cost, run one global search iteration with:

```bash
python optimize.py --run-name global-smoke-001 --mode global --iterations 1
```

Run a domain-specific iteration with, for example:

```bash
python optimize.py --run-name algebra-smoke-001 \
  --mode task-specific --domain algebra --iterations 1
```

One iteration evaluates three generated candidates.  The optimizer only uses
the search split; development selection and final-test evaluation remain
explicit later phases to prevent accidental leakage.

