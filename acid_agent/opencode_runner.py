"""Headless OpenCode runner — the 'hands' of the agent.

We drive `opencode run "<instruction>"` with cwd set to the task workspace.
OpenCode reads/writes files and executes code there; we capture its output.
Atomicity is enforced OUTSIDE opencode: git snapshot before, commit-or-revert after.
"""

import os
import subprocess

from .config import get_settings
from .schemas import ExecResult


def run_opencode(prompt: str, cwd, timeout_s: int | None = None) -> ExecResult:
    s = get_settings()
    cmd = [s.opencode_bin, "run"]
    model = os.environ.get("OPENCODE_MODEL")
    if model:
        cmd += ["--model", model]
    cmd.append(prompt)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s or s.opencode_timeout_s,
        )
        return ExecResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout[-12000:],
            stderr=proc.stderr[-4000:],
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, stdout="", stderr=f"OPENCODE TIMEOUT", returncode=-1)
    except FileNotFoundError:
        return ExecResult(
            ok=False,
            stdout="",
            stderr=f"opencode binary not found at '{s.opencode_bin}'",
            returncode=-1,
        )