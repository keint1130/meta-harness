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

Evaluate the no-retrieval baseline on one domain's search split:

```bash
python inner_loop.py \
  --candidate agents/no_memory.py \
  --split search \
  --domain algebra \
  --output logs/manual/algebra/no_memory/search.json
```

Omit `--domain` to evaluate the global, four-domain pool.  `inner_loop.py`
loads generation and symbolic grading from the sibling `math-frontier-eval`
checkout.  This command calls the configured model and therefore requires
`OPENAI_API_KEY`.

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

