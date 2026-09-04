"""Pydantic models shared across the pipeline."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Decision(BaseModel):
    """One structured analytical decision extracted from evidence."""

    id: str = Field(description="short slug, e.g. 'join_on_station_date'")
    text: str = Field(description="the decision in one sentence")
    rationale: str = Field(description="why this choice, based on evidence")


class Evidence(BaseModel):
    """Consolidated observations from exploration."""

    summary: str = ""
    facts: list[str] = []


class ExecResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class ValidationReport(BaseModel):
    """Outcome of one gate attempt.

    Three-tier like the reference: a component may be `pass`, `watch`, `retry`
    or `skipped`. Only `retry` forces a rollback — `watch` is recorded and lets
    the attempt through, so a soft signal cannot burn a retry budget on its own.
    """

    execution_ok: bool = True
    reflection_ok: bool = True

    # Gating signals, all in raw nats (see confidence.py).
    max_span_surprise: Optional[float] = None    # reference: high => code contradicts evidence
    max_span_divergence: Optional[float] = None  # paper: low  => code not grounded in evidence
    contrast_min_ratio: Optional[float] = None   # low  => evidence prefers the alternative

    # Diagnostic only — recorded, never gates (matches anchor_decision_surprise).
    decision_surprise: Optional[float] = None
    exploration_redundancy: Optional[float] = None

    review_decision: Literal["pass", "watch", "retry"] = "pass"
    components: dict = {}
    watchlist: list[str] = []
    probes: dict = {}

    passed: bool = False
    feedback: str = ""


class MemoryOp(BaseModel):
    op: Literal["insert", "merge", "delete", "split"]
    key: str = Field(description="unique node key")
    content: str = ""
    related_keys: list[str] = []  # edges to create for insert


class MemoryOps(BaseModel):
    """Wrapper so structured output returns a list of memory operations."""

    ops: list[MemoryOp] = []


class UnitResult(BaseModel):
    unit_index: int
    goal: str
    status: Literal["committed", "failed"]
    summary: str = ""
    artifacts: list[str] = []
    git_commit: str = ""