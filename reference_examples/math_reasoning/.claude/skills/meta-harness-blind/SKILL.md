---
name: math-meta-harness-blind
description: Evolve math memory and retrieval harnesses from controlled feedback.
---

# Leakage-safe math harness evolution

Run one proposer iteration entirely inside the supplied working directory. Do
not inspect parent directories, external benchmark files, configs, logs, or
results. Do not run paid evaluations and do not delegate work.

Read `optimizer_state.json`, pooled `anonymous_traces.jsonl`,
`math_harness.py`, and the available candidates under `agents/`. The state says
whether this is a task-specific run and, only then, exposes its top-level
domain. Missing domain/subcategory identities and task boundaries are
intentional and must not be reconstructed from external files.

`anonymous_traces.jsonl` contains detailed search-split feedback only.
Development validation is represented exclusively by aggregate fields in
`optimizer_state.json` (accuracy, token use, and empty count). Do not seek or
reconstruct individual development examples or failures.

Implement exactly the requested number of genuinely different candidates.
Candidates may optimize retrieval, memory representation, example selection,
prompt construction, inferred routing, generation plans, and output handling;
do not merely sweep constants. Never hardcode benchmark problem text, IDs, or
answers.

## Candidate interface

```python
class Candidate:
    def __init__(self, memory, task_context=None): ...
    def build_prompt(self, problem: str) -> str: ...
```

`memory` is a lazy read-only sequence of external solved examples containing
only `problem`, `solution`, `answer`, and `source`. It contains no evaluation
problems or benchmark labels. `task_context` is `{}` for global runs and
`{"domain": ...}` for task-specific runs. `build_prompt` receives only the
current problem statement.

Candidates may import `MathBM25`, `math_tokenize`, and `normalize` from
`math_retriever` when evaluated. Keep imports lazy or avoid constructing a
candidate during syntax checks because loading memory can require network/cache
access.

Write files to `agents/<name>.py`, validate them with `python -m py_compile`,
and write `pending_eval.json` in the current directory:

```json
{
  "iteration": 1,
  "candidates": [
    {
      "name": "snake_case_name",
      "file": "agents/snake_case_name.py",
      "hypothesis": "falsifiable claim",
      "base_system": "source candidate",
      "components": ["memory", "retrieval"]
    }
  ]
}
```
