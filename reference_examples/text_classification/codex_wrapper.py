"""Minimal wrapper around ``codex exec`` with JSONL logging.

The wrapper retains the user's Codex configuration and authentication.
Repository-specific proposer guidance
is injected into the prompt by ``skills``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SessionResult:
    prompt: str
    text: str
    tool_calls: list[dict]
    token_usage: dict
    duration_seconds: float
    model: str
    session_id: str
    exit_code: int
    raw_events: list[dict]
    reasoning_effort: str | None = None
    stderr: str = ""
    log_dir: str | None = None

    def show(self):
        if self.exit_code != 0:
            print(f"  FAILED (exit={self.exit_code})")
            print(f"  {(self.stderr or 'No stderr.')[:300]}")
            return
        for call in self.tool_calls:
            detail = call.get("command") or call.get("path") or ""
            print(f"  tool: {call.get('type', 'tool')}({str(detail)[:120]})")
        text = self.text.strip().replace("\n", " ")
        if text:
            print(f"  text: {text[:200]}")
        print(
            f"  {self.token_usage.get('input_tokens', 0)}in/"
            f"{self.token_usage.get('output_tokens', 0)}out  "
            f"{self.duration_seconds:.1f}s"
        )


def _load_skills(skills):
    contents = []
    for skill in skills or []:
        path = Path(skill)
        if path.is_dir():
            path = path / "SKILL.md"
        if path.is_file():
            contents.append((path.stem, path.read_text(encoding="utf-8")))
    return contents


def _cli_path(path, executable):
    """Translate a WSL mount path when invoking a Windows executable."""
    value = str(path)
    if not str(executable).lower().endswith(".exe"):
        return value
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", value)
    if not match:
        return value
    drive, tail = match.groups()
    return f"{drive.upper()}:\\{tail.replace('/', chr(92))}"


def build_command(prompt, model=None, cwd=None, effort=None, skills=None):
    loaded_skills = _load_skills(skills)
    if loaded_skills:
        skill_text = "\n\n".join(
            f"## Proposer instructions: {name}\n{content}"
            for name, content in loaded_skills
        )
        prompt = f"Follow these proposer instructions:\n\n{skill_text}\n\n{prompt}"

    executable = os.environ.get("CODEX_CLI", "codex")
    cmd = [
        executable,
        "exec",
        "--json",
        "--ephemeral",
        "--config",
        'model_provider="openai"',
        "--ignore-rules",
        "--approve-for-me",
    ]
    if cwd:
        cmd.extend(["--cd", _cli_path(cwd, executable)])
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--config", f'model_reasoning_effort="{effort}"'])
    cmd.append(prompt)
    return cmd


def parse_events(stdout, prompt, model, duration, exit_code, stderr=""):
    events = []
    text = ""
    tool_calls = []
    usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    session_id = ""

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        events.append(event)
        event_type = event.get("type")
        if event_type == "thread.started":
            session_id = event.get("thread_id", "")
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", text)
            elif item.get("type") in {
                "command_execution",
                "file_change",
                "mcp_tool_call",
                "web_search",
            }:
                tool_calls.append(item)
        elif event_type == "turn.completed":
            event_usage = event.get("usage", {})
            for key in usage:
                usage[key] = int(event_usage.get(key, usage[key]) or 0)

    return SessionResult(
        prompt=prompt,
        text=text,
        tool_calls=tool_calls,
        token_usage=usage,
        duration_seconds=duration,
        model=model or "codex-default",
        session_id=session_id,
        exit_code=exit_code,
        raw_events=events,
        stderr=stderr,
    )


def _log_session(result, log_dir, name):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{timestamp}_{name or 'run'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in result.raw_events) + "\n",
        encoding="utf-8",
    )
    (run_dir / "response.md").write_text(result.text, encoding="utf-8")
    meta = {
        "model": result.model,
        "reasoning_effort": result.reasoning_effort,
        "session_id": result.session_id,
        "exit_code": result.exit_code,
        "duration_seconds": round(result.duration_seconds, 2),
        "token_usage": result.token_usage,
        "stderr": result.stderr,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    result.log_dir = str(run_dir)


def run(
    prompt,
    model=None,
    cwd=None,
    log_dir="experience",
    name=None,
    skills=None,
    timeout_seconds=None,
    effort=None,
    **_ignored,
):
    cmd = build_command(
        prompt, model=model, cwd=cwd, effort=effort, skills=skills
    )
    start = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=os.environ.copy(),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout_seconds}s."
    except FileNotFoundError as exc:
        exit_code = 127
        stdout = ""
        stderr = str(exc)

    result = parse_events(
        stdout, prompt, model, time.time() - start, exit_code, stderr
    )
    result.reasoning_effort = effort
    _log_session(result, log_dir, name)
    return result
