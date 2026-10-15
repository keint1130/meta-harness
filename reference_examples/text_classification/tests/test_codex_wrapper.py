import json
import unittest
from unittest import mock

import codex_wrapper


class CodexWrapperTests(unittest.TestCase):
    def test_parse_events_extracts_result_tools_and_usage(self):
        stdout = "\n".join(
            json.dumps(event)
            for event in [
                {"type": "thread.started", "thread_id": "thread-1"},
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": "rg foo"},
                },
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "done"},
                },
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 3,
                        "output_tokens": 4,
                    },
                },
            ]
        )

        result = codex_wrapper.parse_events(stdout, "prompt", None, 1.5, 0)

        self.assertEqual(result.session_id, "thread-1")
        self.assertEqual(result.text, "done")
        self.assertEqual(result.tool_calls[0]["command"], "rg foo")
        self.assertEqual(result.token_usage["input_tokens"], 10)
        self.assertEqual(result.token_usage["output_tokens"], 4)

    def test_build_command_is_automated_and_workspace_scoped(self):
        cmd = codex_wrapper.build_command("do work", cwd="repo", effort="max")

        self.assertEqual(cmd[:3], ["codex", "exec", "--json"])
        self.assertIn("--ephemeral", cmd)
        self.assertNotIn("--ignore-user-config", cmd)
        self.assertIn('model_provider="openai"', cmd)
        self.assertFalse(any(arg.startswith("model_providers.openai.") for arg in cmd))
        self.assertNotIn("--sandbox", cmd)
        self.assertIn("--approve-for-me", cmd)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)

    def test_build_command_accepts_cross_environment_cli_path(self):
        with mock.patch.dict("os.environ", {"CODEX_CLI": "/mnt/c/codex.exe"}):
            cmd = codex_wrapper.build_command(
                "do work", cwd="/mnt/c/Users/test/repo"
            )

        self.assertEqual(cmd[0], "/mnt/c/codex.exe")
        self.assertEqual(cmd[cmd.index("--cd") + 1], r"C:\Users\test\repo")


if __name__ == "__main__":
    unittest.main()
