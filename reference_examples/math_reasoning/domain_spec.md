# Domain Spec: IMO Math Harness Optimization

## Domain Summary

Optimize Python harness code around a frozen `openai/gpt-6-luna` solver for
short-answer IMO-level mathematics.  One evaluation unit is one
IMO-AnswerBench problem.  Model weights, answer grading, and the benchmark
problems are fixed.  Candidate harnesses may change retrieval, example
selection, context construction, prompting, routing, generation control, and
output processing.

The solver uses `reasoning_effort: medium` and a 16,384-token completion budget
so reasoning does not routinely consume the budget before a visible answer.
Sampling temperature is omitted
because GPT-6 does not accept `temperature` when reasoning effort is not
`none`.

The experiment compares four domain-specific harnesses (algebra,
combinatorics, geometry, and number theory) with one global math harness.  The
default smoke budget is one proposer iteration with three candidates.  A full
optimization budget remains to be selected before a paid run.

## Harness and Search Plan

Every new candidate must expose a `Candidate` class with:

```python
class Candidate:
    def __init__(self, memory, task_context=None): ...
    def build_prompt(self, problem: str) -> str: ...
```

`memory` is a lazy read-only external solved-problem corpus.  Its rows are
allowlisted to `problem`, `solution`, `answer`, and `source`; benchmark rows,
labels, and answers are never included.  `task_context` contains the top-level
domain only in task-specific runs and is empty in global runs.  The prediction
method always sees only the current problem statement. Legacy candidates using
`test_problems` remain runnable, but receive only problem text.

The proposer also runs in a controlled view. It receives pooled search traces
and aggregate history without problem IDs or subcategory labels. Global runs
additionally omit top-level domains, task boundaries, and per-domain scores.

The initial baselines are no retrieval, random few-shot retrieval, BM25
retrieval, and the published `evo_geo_solution_indexed` retriever.  Candidate
code may reuse the retrieval primitives and corpora from `math-frontier-eval`.
The proposer should make mechanism-level changes rather than parameter-only
variants.

For task-specific search, run an independent proposer history for each of the
four domains.  For global search, pool all four domains and select candidates
by unweighted macro-average accuracy.  Solver candidate/problem evaluations
must be reported separately from proposer cost.

## Evaluation Plan

IMO-AnswerBench contains 100 problems in each top-level domain.  A deterministic
split assigns 30 search, 20 development holdout, and 50 final-test problems per
domain.  The twelve problems used by the earlier pilot are forced into search,
so none can enter the final test.

- Search: proposer-visible problem traces and iterative candidate evaluation.
- Development validation: evaluated after every candidate and used for the
  outer-loop objective and Pareto frontier. Only aggregate candidate metrics
  are shown to the proposer; no dev problem text, response, answer, ID,
  category, subcategory, or per-problem correctness is exposed.
- Final test: one evaluation after optimization is finalized.

Primary metrics are per-domain accuracy and four-domain macro-average accuracy.
Secondary metrics are input tokens, output tokens, elapsed time, context
length, and API spend when available.  Final comparisons should use paired
problem-level bootstrap confidence intervals and McNemar's test.

The split algorithm and seed are versioned.  Final-test IDs must not appear in
proposer prompts, search logs, or development reports.

## Experience and Logging

For search examples, preserve the problem ID, domain, candidate name, prompt,
model response, normalized score, token totals, elapsed time, and candidate
source revision.  Maintain separate run directories for `global`, `algebra`,
`combinatorics`, `geometry`, and `number_theory`.

The proposer may inspect candidate source, aggregate scores, and search traces.
It must not inspect final-test data.  Development feedback should be aggregate
only.  Each run should keep `evolution_summary.jsonl`, `frontier_val.json`,
candidate source snapshots, and proposer-session metadata.

## Open Questions and Unknowns

- Full paid-search budget: unknown.  Proposed starting point: one iteration,
  three candidates, then estimate cost from measured token use.
- Number of random seeds for final results: unknown.  Proposed minimum: three
  solver seeds or samples per problem if budget permits.
- Category routing is part of the treatment: task-specific harnesses receive
  one run-level oracle domain, while the global harness must infer any useful
  routing from problem text. Subcategory labels are hidden in both conditions.

