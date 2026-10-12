---
name: meta-harness-blind
description: Evolve a general-purpose memory system from aggregate-only feedback.
---

# Dataset-Blind Meta-Harness

Run exactly one iteration of memory-system evolution. Work only in the isolated
working directory supplied by the controller. Do not inspect parent directories
or external config, data, logs, reports, or results. Task identities and
per-task measurements are intentionally hidden.

Do not run benchmarks. The controller evaluates candidates after this session.
Do not delegate to subagents.

## Available evidence

- `optimizer_state.json`: aggregate macro-average accuracy, aggregate context
  length, aggregate Pareto frontier, and aggregate candidate history.
- `anonymous_traces.jsonl`: pooled and deterministically shuffled validation
  traces. Each row contains the system, problem, assembled prompt, prediction,
  ground truth, and correctness, but no task name or task identifier.
- `agents/`: baselines and candidates previously produced in this same blind run.
- `memory_system.py` and `llm.py`: the interfaces candidates must use.

Only draw conclusions supported by this aggregate evidence. Never invent task
identities or per-task behavior. Candidate code, prompts, comments, names, and
hypotheses must remain task-agnostic.

## Required workflow

1. Read `optimizer_state.json`, `anonymous_traces.jsonl`, and the available
   systems in `agents/`. You may infer latent task structure from trace content,
   but must not seek external task identities or per-task scores.
2. Form three falsifiable hypotheses targeting different mechanisms.
3. Prototype each mechanism with pooled trace examples. Try multiple variants
   without separating rows using external dataset information.
4. Implement exactly three genuinely different candidates in `agents/<name>.py`.
5. Check syntax with `python -m py_compile agents/<name>.py`. Do not access a
   parent project or external environment to validate imports.
6. Write `pending_eval.json` in the working directory using the schema below.

Parameter-only variants are not acceptable. Prefer changes to retrieval,
selection, memory representation, prompt structure, or learning strategy.

## Candidate interface

Each candidate must extend `MemorySystem` from `..memory_system` and use
`LLMCallable` from `..llm`. It must implement:

```python
class Candidate(MemorySystem):
    def __init__(self, llm): ...
    def predict(self, input): ...
    def learn_from_batch(self, batch_results): ...
    def get_state(self): ...
    def set_state(self, state): ...
```

Use `self.call_llm(prompt)` and `extract_json_field(response, "final_answer")`.
`predict` must work before learning. `learn_from_batch` receives dictionaries
with `input`, `prediction`, `ground_truth`, `was_correct`, and `metadata`.

## Output

```json
{
  "iteration": 1,
  "candidates": [
    {
      "name": "snake_case_name",
      "file": "agents/snake_case_name.py",
      "hypothesis": "aggregate, falsifiable claim",
      "axis": "exploitation",
      "base_system": "available_system_name",
      "components": ["mechanism_a", "mechanism_b"]
    }
  ]
}
```

Finish with `CANDIDATES: <name1>, <name2>, <name3>`.
