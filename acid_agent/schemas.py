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
    execution_ok: bool = True
    decision_divergence: Optional[float] = None
    max_code_span_divergence: Optional[float] = None
    exploration_redundancy: Optional[float] = None
    reflection_ok: bool = True
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