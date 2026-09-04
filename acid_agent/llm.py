"""Backbone LLM access via the Claude Code CLI.

Every backbone call (planning, decision extraction, reflection, memory
evolution) goes through a headless `claude -p` invocation on the CLI's own
subscription auth. Stateless by design: each call is a fresh session with no
shared context.
"""

import json
import re
import subprocess
import tempfile

from .config import get_settings
from .tracing import traced


def _run_cli(prompt: str, timeout_s: int = 360, isolated: bool = False) -> str:
    """One headless `claude -p` call.

    `isolated=True` makes the call as close to a bare model completion as this
    CLI allows: `--restricted` drops the code-running tools and confines the file
    tools to the working directory, and the working directory is a fresh empty
    temp dir, so there is nothing for the model to read.

    This matters more than it looks. A plain `claude -p` is NOT a tool-free
    completion — it retains Read/Glob/Grep and will happily go find files. A
    denylist is not enough either: `--disallowedTools "Bash,Read,..."` was
    observed routing around the denial through another tool. Only the empty-cwd
    plus `--restricted` combination actually holds.
    """
    s = get_settings()
    from .claude_runner import claude_env

    env = claude_env()
    # Utility calls (plan/decisions/reflect/memory) don't need long reasoning;
    # bounding thinking keeps router round-trips fast and avoids 4-min hangs.
    env.setdefault("MAX_THINKING_TOKENS", "2048")
    args = [s.claude_bin, "-p", prompt, "--output-format", "text"]
    if isolated:
        args += ["--restricted", "--strict-mcp-config"]

    with tempfile.TemporaryDirectory(prefix="acid_isolated_") as sandbox:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=sandbox if isolated else None,
            # Never inherit stdin. When the caller was itself started from a
            # heredoc (`python - <<EOF`), that fd still holds the caller's whole
            # source and a child can seek back to 0 and read it. That leaked
            # seeded CSV rows straight into a supposedly isolated baseline call,
            # which then answered correctly without taking a single action.
            stdin=subprocess.DEVNULL,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[-300:]}")
    return proc.stdout


@traced("backbone.ask", run_type="llm")
def ask(prompt: str) -> str:
    """Plain text completion from the backbone.

    NOTE: this call still has Read/Glob/Grep and runs in the process's cwd (the
    repo root), so it is not a tool-free completion. Use `ask_isolated` where
    the model must reason only from what the caller puts in the prompt.
    """
    return _run_cli(prompt).strip()


@traced("backbone.ask_isolated", run_type="llm")
def ask_isolated(prompt: str) -> str:
    """Text completion with no tool reach: restricted mode in an empty temp cwd.

    Use this whenever the point of the measurement is what the model can do from
    the prompt alone — a baseline arm, or any evaluation where the model finding
    the data itself would invalidate the number.
    """
    return _run_cli(prompt, isolated=True).strip()


@traced(
    "backbone.ask_structured",
    run_type="llm",
    summarize={"schema": lambda s: getattr(s, "__name__", str(s))},
)
def ask_structured(prompt: str, schema):
    """Structured output: ask for JSON matching the schema, parse locally.

    The CLI path has no native function-calling, so we instruct JSON-only
    output, extract the object, and validate against the pydantic schema.
    One retry on CLI or parse failure (timeouts/slow router included).
    """
    schema_json = (
        json.dumps(schema.model_json_schema(), indent=2)
        if hasattr(schema, "model_json_schema")
        else "{}"
    )
    full = (
        prompt
        + "\n\nRespond with ONLY one JSON object matching this JSON Schema "
        "(no markdown, no commentary):\n" + schema_json
    )
    last_err = None
    for attempt in range(2):
        try:
            raw = _run_cli(full)
        except Exception as e:  # TimeoutExpired / CLI rc != 0 → retry once
            last_err = e
            continue
        try:
            return schema.model_validate_json(_extract_json(raw))
        except Exception as e:
            last_err = e
    raise RuntimeError(f"structured output failed after retry: {last_err}")


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text.strip()
