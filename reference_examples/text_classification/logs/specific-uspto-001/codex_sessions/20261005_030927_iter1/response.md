Implemented and import-validated three candidates:

- [correction_ledger.py](C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/agents/correction_ledger.py) keeps recent failed cases and their corrections.
- [scaffold_memory.py](C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/agents/scaffold_memory.py) factors shared input text while preserving exact reconstruction.
- [lesson_window_memory.py](C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/agents/lesson_window_memory.py) distills feedback windows into LLM-authored lessons and retains recent cases.

Prototypes used the 48 saved training examples. Scaffold factoring reduced the demonstration section from 8,264 to 6,572 characters with exact reconstruction. The lesson-window comparison supported a window of eight. The logs lack per-example predictions, so correction flags in those prototypes were simulated. No benchmark was run.

The manifest is at [pending_eval.json](C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/logs/specific-uspto-001/pending_eval.json). No post-eval report was due: this run has no `evolution_summary.jsonl`. `config.yaml` remains modified in the worktree; I left it untouched.

CANDIDATES: correction_ledger, scaffold_memory, lesson_window_memory