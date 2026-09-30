import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from text_classification.llm import LLM
from litellm.utils import get_optional_params


class ReasoningEffortTests(unittest.TestCase):
    def test_real_litellm_parameter_validation(self):
        for effort in ("medium", "none"):
            with self.subTest(effort=effort), LLM(
                model="openai/gpt-6-luna", reasoning_effort=effort, temperature=0.0
            ) as llm:
                params = get_optional_params(
                    model="gpt-6-luna", custom_llm_provider="openai",
                    **llm._model_kwargs,
                )
                self.assertEqual(params["reasoning_effort"], effort)
                if effort == "none":
                    self.assertEqual(params["temperature"], 0.0)
                else:
                    self.assertNotIn("temperature", params)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        cache = patch("text_classification.llm.CACHE_DIR", Path(temporary.name))
        cache.start()
        self.addCleanup(cache.stop)

    def test_effort_reaches_completion_and_changes_cache_key(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )
        with LLM(model="openai/gpt-6-luna", reasoning_effort="medium") as llm:
            with patch("text_classification.llm.litellm_completion", return_value=response) as call:
                llm._provider._call_completion("question", None, llm._model_kwargs)
            self.assertEqual(call.call_args.kwargs["reasoning_effort"], "medium")
            self.assertEqual(llm.get_usage()["reasoning_effort"], "medium")
            key = llm._provider._cache_path("question", None, llm._model_kwargs)
            changed = dict(llm._model_kwargs, reasoning_effort="high")
            self.assertNotEqual(key, llm._provider._cache_path("question", None, changed))
