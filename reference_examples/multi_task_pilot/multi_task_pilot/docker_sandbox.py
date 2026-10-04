from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def resolve_docker_executable() -> str:
    discovered = shutil.which("docker")
    if discovered:
        return discovered
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = (
            Path(local_app_data)
            / "Programs"
            / "DockerDesktop"
            / "resources"
            / "bin"
            / "docker.exe"
        )
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Docker CLI was not found. Restart the terminal after installing Docker Desktop."
    )


@dataclass(frozen=True)
class DockerSandbox:
    docker_executable: str | None = None
    image: str = "python:3.13-alpine"
    timeout_seconds: int = 10
    memory: str = "256m"
    cpus: float = 1.0
    pids_limit: int = 64

    def command(self, workspace: Path) -> list[str]:
        source = workspace.resolve().as_posix()
        return [
            self.docker_executable or resolve_docker_executable(),
            "run",
            "--interactive",
            "--rm",
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            str(self.pids_limit),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--mount",
            f"type=bind,source={source},target=/workspace,readonly",
            self.image,
            "python",
            "-I",
            "/workspace/runner.py",
        ]

    def run_files(
        self, files: dict[str, str], *, stdin: str = ""
    ) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="moharness-code-") as tmp:
            workspace = Path(tmp)
            if "runner.py" not in files:
                raise ValueError("files must contain runner.py")
            for relative_name, contents in files.items():
                relative = Path(relative_name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe sandbox file name: {relative_name!r}")
                destination = workspace / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(contents, encoding="utf-8")
            command = self.command(workspace)
            child_env = os.environ.copy()
            docker_bin = str(Path(command[0]).resolve().parent)
            child_env["PATH"] = docker_bin + os.pathsep + child_env.get("PATH", "")
            try:
                completed = subprocess.run(
                    command,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    env=child_env,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    exit_code=124,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    timed_out=True,
                )
        return SandboxResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def run_python(self, source: str, stdin: str = "") -> SandboxResult:
        return self.run_files({"runner.py": source}, stdin=stdin)
