---
name: math-meta-harness
description: Propose one iteration of math harness candidates.
---

# Math Harness Evolution

Implement exactly three new candidate harnesses in `agents/`.  Each file must
expose `Candidate(test_problems=None)` with `build_prompt(problem: str) -> str`.

Read `domain_spec.md`, `config.yaml`, existing candidates, and the run files
identified in the task prompt.  Analyze `evolution_summary.jsonl` and search
traces when present.  Never inspect or use the final-test split.  Do not run
paid model evaluations; the outer loop does that.

Candidates may import retrieval primitives from the sibling
`math-frontier-eval` repository (`math_retriever`) and may change retrieval,
routing, example selection, context construction, prompting, or output
processing.  Prefer mechanism changes over numeric parameter variants.  Do not
hardcode benchmark problem IDs, answers, or problem text.

Import-check each candidate without constructing it.  Then write
`pending_eval.json` at the path stated in the task prompt:

```json
{
  "iteration": 1,
  "candidates": [
    {
      "name": "candidate_name",
      "file": "agents/candidate_name.py",
      "hypothesis": "falsifiable hypothesis",
      "base_system": "source candidate",
      "components": ["retrieval", "prompting"]
    }
  ]
}
```

