"""Claude Code CLI adapter — execution backbone for both agent arms.

Headless one-shot tasks:

    claude -p "<task>" --output-format text --dangerously-skip-permissions

Auth is whatever the `claude` CLI is logged into (Claude Code subscription);
nothing here injects an API key or a third-party base URL.
"""

import os
import subprocess

from .config import get_settings
from .llm import strip_ansi
from .schemas import ExecResult
from .tracing import traced


def claude_env() -> dict:
    """Environment for every claude CLI invocation.

    Inherited ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN are dropped so a stale
    export can't silently redirect the CLI away from the subscription login.
    """
    s = get_settings()
    env = os.environ.copy()
    env.pop("ANTHROPIC_BASE_URL", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    if s.claude_model:
        env["ANTHROPIC_MODEL"] = s.claude_model
    return env


@traced("claude_code.session", run_type="llm", summarize={"cwd": str})
def run_claude(prompt: str, cwd, timeout_s: int | None = None) -> ExecResult:
    """Run one headless agent session with `prompt` in the given directory.

    With backbone=claude this is a Claude Code session (`--dangerously-skip-
    permissions`); with backbone=opencode it is an opencode agent session
    (`build` agent) on the opencode go subscription — the same harness role,
    whichever gateway is active.
    """
    s = get_settings()
    if s.backbone == "opencode":
        return _run_opencode_session(prompt, cwd, timeout_s or s.claude_timeout_s)
    cmd = [
        s.claude_bin,
        "-p", prompt,
        "--output-format", "text",
        "--dangerously-skip-permissions",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s or s.claude_timeout_s,
            env=claude_env(),
            # See llm._run_cli: an inherited stdin can still hold the caller's
            # own source (heredoc-launched evals) and leak it into the session.
            stdin=subprocess.DEVNULL,
        )
        return ExecResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout[-12000:],
            stderr=proc.stderr[-4000:],
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, stdout="", stderr="CLAUDE TIMEOUT", returncode=-1)
    except FileNotFoundError:
        return ExecResult(
            ok=False, stdout="", stderr=f"claude binary not found at '{s.claude_bin}'", returncode=-1
        )


def _run_opencode_session(prompt: str, cwd, timeout_s: int) -> ExecResult:
    """One headless opencode agent session — the harness role of run_claude
    on the opencode gateway. `--dir` roots the agent in the workspace; headless
    `run` executes tools without interactive permission prompts."""
    s = get_settings()
    args = [s.opencode_bin, "run", "-m", s.opencode_model]
    if cwd:
        args += ["--dir", str(cwd)]
    try:
        proc = subprocess.run(
            args + [prompt],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd) if cwd else None,
            env={k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")},
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


def check_claude_available() -> bool:
    from shutil import which

    return which(get_settings().claude_bin) is not None
