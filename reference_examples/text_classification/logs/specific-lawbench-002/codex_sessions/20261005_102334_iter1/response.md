Created three task-agnostic candidates and listed them in [pending_eval.json](/C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/logs/specific-lawbench-002/proposer_view/text_classification/pending_eval.json):

- [bm25_retrieval.py](/C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/logs/specific-lawbench-002/proposer_view/text_classification/agents/bm25_retrieval.py) — query-ranked examples within a fixed context budget.
- [contrastive_replay.py](/C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/logs/specific-lawbench-002/proposer_view/text_classification/agents/contrastive_replay.py) — retrieves similar prior errors with their corrections.
- [label_prototypes.py](/C:/Users/keinf/vscode/MoHarness/meta-harness/reference_examples/text_classification/logs/specific-lawbench-002/proposer_view/text_classification/agents/label_prototypes.py) — presents compact, query-supported answer profiles.

All three passed `py_compile`, and the manifest parsed with exactly three candidates. No benchmarks were run.

CANDIDATES: bm25_retrieval, contrastive_replay, label_prototypes