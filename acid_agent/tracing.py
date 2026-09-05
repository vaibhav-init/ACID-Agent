"""LangSmith tracing wiring.

Two halves of this app trace very differently:

  * LangGraph invocations (task graph, unit graph) are instrumented for free by
    langchain-core once LANGSMITH_TRACING is in the OS environment.
  * The backbone LLM calls are NOT. They are headless CLI subprocesses, invisible
    to LangChain callbacks, so a trace would show empty graph nodes with no
    prompts, no outputs and no timing attribution. `@traced` below annotates
    them so the plan / decisions / codegen / reflection calls show up as real
    LLM spans.

Settings stays the single config surface: `configure()` pushes the LANGSMITH_*
values from .env into os.environ, which is the only thing the SDK reads.
"""

import os
from typing import Any, Callable

from .config import get_settings


def configure() -> bool:
    """Export LangSmith env vars from Settings. Returns True if tracing is on.

    Called at package import so any entry point (CLI, pytest, scripts/) gets the
    same behavior without each one remembering to opt in.
    """
    s = get_settings()
    enabled = bool(s.langsmith_tracing and s.langsmith_api_key)
    # Written either way: an api-key-less LANGSMITH_TRACING=true otherwise makes
    # every call retry against a 401 and slows the run down for nothing.
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        if s.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    return enabled


def _redactor(drop: tuple[str, ...], summarize: dict[str, Callable[[Any], Any]]):
    """Build a process_inputs hook that keeps trace payloads small and legible."""

    def process(inputs: dict) -> dict:
        out = {k: v for k, v in inputs.items() if k not in drop}
        for key, fn in summarize.items():
            if key in out:
                try:
                    out[key] = fn(out[key])
                except Exception:
                    out[key] = repr(out[key])[:200]
        return out

    return process


def traced(
    name: str,
    run_type: str = "chain",
    drop: tuple[str, ...] = (),
    summarize: dict[str, Callable[[Any], Any]] | None = None,
):
    """@traceable with input redaction, degrading to a no-op without langsmith.

    `drop` removes arguments that would bloat the trace (seeded file contents,
    which can be megabytes of xlsx bytes); `summarize` rewrites ones that don't
    serialize usefully (a pydantic class object -> its name).
    """
    try:
        from langsmith import traceable
    except ImportError:  # tracing is optional; never break a run over it
        return lambda fn: fn

    return traceable(
        name=name,
        run_type=run_type,
        process_inputs=_redactor(drop, summarize or {}),
    )
