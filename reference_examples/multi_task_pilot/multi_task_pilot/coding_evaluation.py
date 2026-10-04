from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .contracts import EvalCase, TaskFamily
from .docker_sandbox import DockerSandbox


_RUNNER = r"""
import base64
import json
import os
import pickle
import subprocess
import sys
import zlib


def decode_private(encoded):
    # The upstream format is trusted benchmark data, but pickle is still
    # decoded only inside this disposable, networkless container.
    raw = zlib.decompress(base64.b64decode(encoded))
    return json.loads(pickle.loads(raw))


def normalized(value):
    if isinstance(value, tuple):
        return [normalized(item) for item in value]
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    return value


def expected_value(text):
    try:
        return json.loads(text)
    except Exception:
        return text.strip()


def functional(test, func_name):
    arguments = [json.loads(line) for line in test["input"].splitlines()]
    driver = '''
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location("candidate", "/workspace/solution.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
target = module.Solution() if hasattr(module, "Solution") else module
fn = getattr(target, sys.argv[1])
args = json.loads(sys.stdin.read())
print(json.dumps(fn(*args), default=str))
'''
    path = "/tmp/call.py"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(driver)
    completed = subprocess.run(
        [sys.executable, "-I", path, func_name],
        input=json.dumps(arguments), text=True, capture_output=True, timeout=4,
        cwd="/tmp",
    )
    if completed.returncode:
        return False, completed.stderr[-1000:]
    try:
        actual = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return False, "non-JSON function output: " + completed.stdout[-500:]
    return normalized(actual) == normalized(expected_value(test["output"])), ""


def stdin_case(test):
    completed = subprocess.run(
        [sys.executable, "-I", "/workspace/solution.py"],
        input=test["input"], text=True, capture_output=True, timeout=4,
        cwd="/tmp",
    )
    if completed.returncode:
        return False, completed.stderr[-1000:]
    actual = "\n".join(line.rstrip() for line in completed.stdout.strip().splitlines())
    expected = "\n".join(line.rstrip() for line in test["output"].strip().splitlines())
    if actual == expected:
        return True, ""
    try:
        return float(actual) == float(expected), ""
    except Exception:
        return False, "expected=" + repr(expected) + "; actual=" + repr(actual)


payload = json.loads(sys.stdin.read())
tests = list(payload["public_tests"])
if payload.get("private_tests_encoded"):
    tests.extend(decode_private(payload["private_tests_encoded"]))
func_name = payload.get("metadata", {}).get("func_name")
failures = []
for index, test in enumerate(tests):
    try:
        if test.get("testtype") == "functional" or func_name:
            passed, detail = functional(test, func_name)
        else:
            passed, detail = stdin_case(test)
    except subprocess.TimeoutExpired:
        passed, detail = False, "per-test timeout"
    except Exception as exc:
        passed, detail = False, type(exc).__name__ + ": " + str(exc)
    if not passed:
        failures.append({"index": index, "detail": detail})
        break
print(json.dumps({"passed": not failures, "tests": len(tests), "failures": failures}))
"""


@dataclass(frozen=True)
class CodingEvaluation:
    passed: bool
    tests: int
    detail: str


def extract_python_code(answer: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.DOTALL | re.I)
    return blocks[-1].strip() if blocks else answer.strip()


def evaluate_livecodebench(
    case: EvalCase, answer: str, sandbox: DockerSandbox
) -> CodingEvaluation:
    if case.task.family is not TaskFamily.CODING:
        raise ValueError("Expected a coding case")
    payload = {
        "public_tests": case.task.metadata.get("public_test_cases", []),
        "private_tests_encoded": case.reference["private_test_cases_encoded"],
        "metadata": case.reference.get("metadata", {}),
    }
    result = sandbox.run_files(
        {"runner.py": _RUNNER, "solution.py": extract_python_code(answer)},
        stdin=json.dumps(payload),
    )
    if result.timed_out:
        return CodingEvaluation(False, 0, "container timeout")
    if result.exit_code:
        return CodingEvaluation(False, 0, result.stderr[-2000:])
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return CodingEvaluation(False, 0, "invalid evaluator output: " + result.stdout[-1000:])
    return CodingEvaluation(
        bool(report["passed"]),
        int(report["tests"]),
        json.dumps(report.get("failures", [])),
    )
