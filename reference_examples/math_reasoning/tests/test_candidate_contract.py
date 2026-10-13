from pathlib import Path

import pytest

from inner_loop import load_candidate


def test_no_memory_candidate_contract():
    path = Path(__file__).parents[1] / "agents" / "no_memory.py"
    candidate = load_candidate(path)(test_problems=[])
    prompt = candidate.build_prompt("Find x.")
    assert "Find x." in prompt
    assert "\\boxed{}" in prompt


def test_candidate_requires_candidate_class(tmp_path):
    path = tmp_path / "invalid.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Candidate"):
        load_candidate(path)

