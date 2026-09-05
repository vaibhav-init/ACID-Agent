"""Backbone LLM access via a headless CLI gateway.

Every backbone call (planning, decision extraction, reflection, memory
evolution) goes through headless `opencode run -m <model>` on the opencode
go subscription (the Qwen model family — the paper's backbone).

Stateless by design: each call is a fresh session with no shared context.
"""

import json
import os
import re
import subprocess
import tempfile

from .config import get_settings
from .tracing import traced

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def strip_ansi(text: str) -> str:
    """Remove ANSI escapes; opencode decorates its streams with them."""
    return _ANSI.sub("", text)


def _run_opencode(prompt: str, timeout_s: int, isolated: bool) -> str:
    """One headless `opencode run` call.

    stdout carries only the assistant's response; the status line and ANSI
    decoration go to stderr. Non-isolated calls run in the process cwd with
    full agent tools. Isolated calls use the read-only `plan` agent inside a
    fresh empty temp dir — nothing for the model to read but the prompt.
    """
    s = get_settings()
    args = [s.opencode_bin, "run", "-m", s.opencode_model]
    if isolated:
        args += ["--agent", "plan"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    with tempfile.TemporaryDirectory(prefix="acid_isolated_") as sandbox:
        proc = subprocess.run(
            args + [prompt],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=sandbox if isolated else None,
            # Never inherit stdin: it can hold the caller's own source
            # (heredoc-launched evals), which a child could read back.
            stdin=subprocess.DEVNULL,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"opencode CLI failed (rc={proc.returncode}): {proc.stderr[-300:]}")
    return strip_ansi(proc.stdout)


def _run_cli(prompt: str, timeout_s: int = 360, isolated: bool = False) -> str:
    """One headless backbone call.

    `isolated=True` makes the call as close to a bare model completion as the
    CLI allows: the read-only `plan` agent in a fresh empty temp cwd, so the
    model can reason only from what the caller puts in the prompt. This
    matters for baselines and evaluation: a model that can go find the data
    itself would invalidate the number.
    """
    return _run_opencode(prompt, timeout_s=timeout_s, isolated=isolated)


@traced("backbone.ask", run_type="llm")
def ask(prompt: str) -> str:
    """Plain text completion from the backbone.

    NOTE: this call still has full agent tools and runs in the process's cwd
    (the repo root), so it is not a tool-free completion. Use `ask_isolated`
    where the model must reason only from what the caller puts in the prompt.
    """
    return _run_cli(prompt).strip()


@traced("backbone.ask_isolated", run_type="llm")
def ask_isolated(prompt: str) -> str:
    """Text completion with no tool reach: read-only agent in an empty temp cwd.

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
