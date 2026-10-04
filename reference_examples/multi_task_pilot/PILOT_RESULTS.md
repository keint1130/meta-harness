# Pilot results

## Fixed development baselines

All scores use 30 deterministic development examples per family. The held-out
50-example test splits have not been evaluated.

| Harness | Coding | Retrieval | MATH-500 Level 5 | Macro |
|---|---:|---:|---:|---:|
| Shared baseline | 14/30 (46.7%) | 19/30 (63.3%) | 25/30 (83.3%) | 64.4% |

The MATH score above uses the corrected evaluator. See
`MATH_EVALUATOR_AUDIT.md` for the false-negative audit and difficulty change.

## Constrained optimization: one proposal per track

The model, reasoning effort, dataset, LLM/tool budgets, and sandbox limits were
fixed. Codex could change only a declarative harness specification containing a
system prompt and family instructions.

| Track | Candidate score | Frontier score | Accepted |
|---|---:|---:|---:|
| Coding-specific | 13/30 (43.3%) | 14/30 (46.7%) | No |
| Retrieval-specific | 16/30 (53.3%) | 19/30 (63.3%) | No |
| Math-specific | 25/30 (83.3%) | 25/30 (83.3%) | No |
| Global | 61/90, macro 67.8% | macro 64.4% | Yes |

The accepted global candidate scored 16/30 coding, 18/30 retrieval, and 27/30
math. This is a development result from one proposal and one run, not evidence
of a stable advantage. No held-out score should be computed until the search
budget, frontier, and evaluation protocol are frozen.

## Methodology note

The first proposer reports accidentally included the encoded LiveCodeBench
development private-test payload inside failure records. The generated
candidate specifications were scanned and contain no task IDs, private-test
content, or checked answers; they contain only general instructions. The report
builder has nevertheless been corrected: coding references now expose only
pass/fail feedback, tool source/output is truncated, and raw private tests stay
evaluator-only.

## Sanitized confirmation run

Coding-specific and global candidates were independently regenerated from the
sanitized reports, starting from the baseline rather than an earlier candidate.

| Track | Sanitized candidate | Baseline | Accepted |
|---|---:|---:|---:|
| Coding-specific | 13/30 (43.3%) | 14/30 (46.7%) | No |
| Global | 56/90, macro 62.2% | 58/90, macro 64.4% | No |

The sanitized global result was 14/30 coding, 17/30 retrieval, and 25/30 math.
Both sanitized frontier files are byte-identical to the baseline harness. Thus,
the earlier 67.8% global result is retained only as an exploratory result and
must not be treated as the confirmed frontier. Under the valid one-proposal
comparison, no track improved over the initial harness.

## Held-out test

All valid sanitized frontiers remained byte-identical to the baseline harness,
so the shared frontier was evaluated once on the frozen 50-example test split
for each family. Repeating the identical harness four times would add sampling
noise rather than compare different frontiers.

| Family | Correct | Accuracy |
|---|---:|---:|
| Coding | 34/50 | 68.0% |
| Retrieval | 29/50 | 58.0% |
| MATH-500 Level 5 | 49/50 | 98.0% |
| Equal-family macro | — | 74.7% |

The original test run recorded MATH as 48/50 because one request failed with a
600-second API connection timeout. That single task was rerun once with the
same frozen harness, model, settings, and manifest; it scored correctly. The
original run remains unchanged under `logs/final-heldout-baseline/`, and the
retry is separately preserved under `logs/final-heldout-timeout-retry/`.

The remaining MATH error was manually checked against the official solution.
It was a substantive branch/domain error involving inverse cotangent, not an
answer-extraction failure. The large coding difference between development
(46.7%) and test (68.0%) also indicates material subset variance at these small
sample sizes, so results should be reported with counts and split sizes rather
than treated as precise population estimates.
