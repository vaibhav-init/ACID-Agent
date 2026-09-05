"""Headless agent-session runner — the execution backbone for both agent arms.

One session = one `opencode run` on the opencode go subscription:
the `build` agent executes tools headless in the given directory.
Auth is the CLI's own subscription login; nothing here injects an API key.
"""

import os
import subprocess
from pathlib import Path

from .config import get_settings
from .llm import strip_ansi
from .schemas import ExecResult
from .tracing import traced


@traced("opencode.session", run_type="llm", summarize={"cwd": str})
def run_opencode_session(prompt: str, cwd, timeout_s: int | None = None) -> ExecResult:
    """One headless opencode agent session with `prompt` in the given directory.

    `--dir` roots the agent in the workspace; headless `run` executes tools
    without interactive permission prompts.
    """
    s = get_settings()
    # opencode resolves --dir from its own internal cwd (client/server), so a
    # relative workspace path fails with "Failed to change directory" even when
    # the subprocess cwd is already correct. Always hand it an absolute path.
    root = Path(cwd).resolve() if cwd else None
    args = [s.opencode_bin, "run", "-m", s.opencode_model]
    if root:
        args += ["--dir", str(root)]
    try:
        proc = subprocess.run(
            args + [prompt],
            capture_output=True,
            text=True,
            timeout=timeout_s or s.session_timeout_s,
            cwd=str(root) if root else None,
            env={k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")},
            # An inherited stdin can hold the caller's own source (heredoc-launched
            # evals); a child must never be able to seek back and read it.
            stdin=subprocess.DEVNULL,
        )
        return ExecResult(
            ok=proc.returncode == 0,
            stdout=strip_ansi(proc.stdout)[-12000:],
            stderr=proc.stderr[-4000:],
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, stdout="", stderr="OPENCODE TIMEOUT", returncode=-1)
    except FileNotFoundError:
        return ExecResult(
            ok=False, stdout="", stderr=f"opencode binary not found at '{s.opencode_bin}'", returncode=-1
        )


def check_opencode_available() -> bool:
    from shutil import which

    return which(get_settings().opencode_bin) is not None
