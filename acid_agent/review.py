"""Interpretation-review mechanisms — the reactive port and the proactive addition.

Failure mode targeted (measured on archeology-hard-7, 2026-09-04): a unit commits
to a wrong *reading* of an ambiguous goal ("within 0.1 degrees" as a box instead
of Euclidean distance) and every downstream component approves it, because they
all check evidence↔code *consistency*, not method *correctness*. Six runs, six
identical wrong answers.

Two mechanisms, deliberately different in kind:

1. REACTIVE — `AnswerCandidateTracker`, adapted from the reference's
   `da_agent/review/answer_candidate.py`. Registers one answer candidate per
   attempt from the execution output; when two DIFFERENT candidates exist, the
   next attempt's prompts get a comparison block that says: choose on method,
   not recency. The reference registers candidates only at Terminate steps; here
   the per-attempt clean re-run (`ws.run_script`) is the candidate source, which
   maps naturally onto our retry loop. Limitation (kept from the reference):
   it is a *disagreement* detector — a consistently wrong pipeline produces a
   single candidate and never triggers it.

2. PROACTIVE — the interpretation probe. No reference equivalent exists: before
   first codegen, the backbone must enumerate ambiguous terms in the goal, name
   the alternative readings, judge whether the chosen reading is the standard
   one, and output revised decisions when it is not. This attacks the exact
   class the reactive mechanism is blind to: the confident first reading.
"""

import re

from pydantic import BaseModel, Field

from .schemas import Decision

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def last_number(text: str) -> float | None:
    """The last numeric token in an execution output — the usual final result."""
    m = _NUMBER.findall(text.replace(",", ""))
    return float(m[-1]) if m else None


class AnswerCandidateTracker:
    """Registers per-attempt answer candidates and detects disagreements."""

    def __init__(self) -> None:
        self.candidates: list[dict] = []

    def register(self, stdout: str, attempt: int, method: str = "") -> float | None:
        val = last_number(stdout)
        if val is not None:
            self.candidates.append({"value": val, "attempt": attempt, "method": method[:120]})
        return val

    def distinct_values(self) -> list[float]:
        out: list[float] = []
        for c in self.candidates:
            if c["value"] not in out:
                out.append(c["value"])
        return out

    def comparison_block(self) -> str | None:
        """Vendor-style comparison prompt; None until two candidates disagree."""
        vals = self.distinct_values()
        if len(vals) < 2:
            return None
        lines = chr(10).join(
            f"- attempt {c['attempt']}: {c['value']!r}" + (f"  (via: {c['method']})" if c["method"] else "")
            for c in self.candidates
        )
        return f"""# ANSWER CANDIDATE COMPARISON #
Different final-answer candidates have appeared across attempts at this unit. Do NOT
choose based on recency. Compare the computation paths and METHOD choices — definitions,
filters, units, distance metrics, inclusive/exclusive bounds, tie-handling — before
relying on either answer:

{lines}

If a prior candidate used a non-standard reading of the goal, prefer the standard one
and say why in one sentence."""


def candidate_block(candidates: list[dict]) -> str | None:
    """Stateless form of the tracker (candidates arrive from graph state)."""
    t = AnswerCandidateTracker()
    t.candidates = list(candidates or [])
    return t.comparison_block()


class InterpretationReview(BaseModel):
    """Result of the proactive ambiguity probe over a unit goal + decisions."""

    ambiguous_terms: list[str] = Field(default_factory=list, description="terms in the goal with more than one reasonable reading")
    analysis: str = Field(default="", description="which reading each decision takes, and the alternatives")
    chosen_reading_standard: bool = Field(default=True, description="true if the decisions use the reading a domain expert would consider standard")
    revised_decisions: list[Decision] = Field(default_factory=list, description="full replacement decisions, only when the chosen reading is non-standard")


INTERPRETATION_PROMPT = """Ambiguity review before implementation.

Task: {task}
Unit goal: {goal}
Evidence summary: {evidence}
Current decisions:
{decisions}

1) List every term in the goal that admits more than one reasonable interpretation
   (distance metrics, units, inclusive/exclusive bounds, "year" vs "record", which
   data source a phrase refers to, tie-handling, rounding, etc.).
2) For each, state which reading the current decisions take and what the
   alternative readings are.
3) Judge whether the chosen reading is the one a domain expert would consider the
   standard/default interpretation of this question as asked. If ANY decision
   takes a non-standard reading, output the FULL revised decision list (same
   decisions otherwise)."""


def apply_interpretation_review(decisions: list[str], probe: InterpretationReview) -> list[str]:
    """Adopt the probe's revisions only when it judges the reading non-standard.

    A revision without the non-standard flag is noise, not signal — the probe
    was asked to revise ONLY in that case.
    """
    if probe.chosen_reading_standard or not probe.revised_decisions:
        return decisions
    revised = [d.text for d in probe.revised_decisions]
    return revised if revised else decisions
