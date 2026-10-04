from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from litellm import completion as litellm_completion

from .contracts import HarnessResult, Task, ToolSet


SYSTEM_PROMPT = """You are a general problem-solving agent. Solve the user's task accurately.
You may use retrieve for frozen evidence and python_exec for calculation or debugging.
Use tools only when useful. Respect the requested final-answer format. For coding tasks,
return a complete solution in one Python code block."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Run Python in an isolated, networkless Docker container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "stdin": {"type": "string"},
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Search the frozen documents supplied with this task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
        },
    },
]


def _as_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    result = {"role": "assistant", "content": getattr(message, "content", None)}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            call.model_dump(exclude_none=True) if hasattr(call, "model_dump") else call
            for call in tool_calls
        ]
    return result


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = vars(usage)
    return {
        key: int(usage.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


class BaselineHarness:
    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        max_llm_calls: int = 3,
        completion_fn: Callable[..., Any] = litellm_completion,
        system_prompt: str = SYSTEM_PROMPT,
        family_instructions: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_llm_calls = max_llm_calls
        self.completion_fn = completion_fn
        self.system_prompt = system_prompt
        self.family_instructions = family_instructions or {}

    def solve(self, task: Task, tools: ToolSet) -> HarnessResult:
        instruction = self.family_instructions.get(task.family.value, "")
        system_prompt = self.system_prompt
        if instruction:
            system_prompt += f"\n\nTask-family instruction:\n{instruction}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.prompt},
        ]
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answer = ""
        llm_calls = 0
        for _ in range(self.max_llm_calls):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
            }
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort
            response = self.completion_fn(**kwargs)
            llm_calls += 1
            for key, value in _usage(response).items():
                totals[key] += value
            message = response.choices[0].message
            message_dict = _as_dict(message)
            messages.append(message_dict)
            answer = message_dict.get("content") or ""
            tool_calls = message_dict.get("tool_calls") or []
            if not tool_calls:
                break
            for call in tool_calls:
                function = call["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                try:
                    if function["name"] == "python_exec":
                        output = tools.python_exec(
                            arguments["source"], arguments.get("stdin", "")
                        )
                    elif function["name"] == "retrieve":
                        output = tools.retrieve(
                            arguments["query"], int(arguments.get("top_k", 5))
                        )
                    else:
                        output = f"Unknown tool: {function['name']}"
                except RuntimeError as exc:
                    # A budget limit is an observation for the agent, not an
                    # infrastructure failure for the benchmark process.
                    output = f"Tool unavailable: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(output, ensure_ascii=False),
                    }
                )
        return HarnessResult(
            answer=answer,
            tool_calls=list(getattr(tools, "calls", [])),
            usage=totals,
            metadata={"llm_calls": llm_calls},
        )


def load_harness_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"name", "description", "system_prompt", "family_instructions"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown harness spec fields: {sorted(unknown)}")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("Harness spec requires a non-empty name")
    if not isinstance(spec.get("system_prompt"), str) or not spec["system_prompt"].strip():
        raise ValueError("Harness spec requires a non-empty system_prompt")
    instructions = spec.get("family_instructions", {})
    if not isinstance(instructions, dict):
        raise ValueError("family_instructions must be an object")
    valid_families = {"coding", "retrieval", "math"}
    if set(instructions) - valid_families:
        raise ValueError("family_instructions contains an unknown task family")
    if any(not isinstance(value, str) for value in instructions.values()):
        raise ValueError("Every family instruction must be a string")
    return spec
