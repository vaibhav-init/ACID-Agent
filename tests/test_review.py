"""Tests for the interpretation-review mechanisms (acid_agent/review.py).

Covers the reactive port (AnswerCandidateTracker semantics) and the proactive
probe application rule, plus the unit_graph wiring for both.
"""

import uuid

import pytest

from acid_agent import review
from acid_agent.review import (
    AnswerCandidateTracker,
    InterpretationReview,
    apply_interpretation_review,
    candidate_block,
    last_number,
)


# --- candidate extraction ------------------------------------------------

def test_last_number_picks_final_value():
    assert last_number("processed 1388 rows\nflagged: 295") == 295.0


def test_last_number_handles_negatives_and_commas():
    assert last_number("difference = -0.008003975") == -0.008003975
    assert last_number("total was 1,234 units") == 1234.0


def test_last_number_none_without_digits():
    assert last_number("no numeric output") is None


# --- reactive tracker ------------------------------------------------------

def test_tracker_silent_until_two_candidates_disagree():
    t = AnswerCandidateTracker()
    t.register("flagged: 295", attempt=1, method="count within 0.1 deg")
    assert t.comparison_block() is None  # single candidate: nothing to compare
    t.register("flagged: 295", attempt=2)
    assert t.comparison_block() is None  # same value: no disagreement
    t.register("flagged: 274", attempt=3)
    block = t.comparison_block()
    assert block is not None
    assert "Do NOT" in block and "295" in block and "274" in block
    assert "recency" in block


def test_candidate_block_stateless_form():
    assert candidate_block([]) is None
    assert candidate_block([{"value": 1.0, "attempt": 1, "method": "m"}]) is None
    b = candidate_block([
        {"value": 1.0, "attempt": 1, "method": "m"},
        {"value": 2.0, "attempt": 2, "method": "m"},
    ])
    assert b is not None and "attempt 1" in b and "attempt 2" in b


# --- proactive probe ---------------------------------------------------------

def _probe(standard=True, revised_texts=None):
    from acid_agent.schemas import Decision
    revised = [Decision(id=f"r{i}", text=t, rationale="probe") for i, t in enumerate(revised_texts or [])]
    return InterpretationReview(
        ambiguous_terms=["within 0.1 degrees"],
        analysis="box vs euclidean",
        chosen_reading_standard=standard,
        revised_decisions=revised,
    )


def test_probe_revision_applied_only_when_reading_nonstandard():
    decisions = ["use box (Chebyshev) neighborhood"]
    revised = ["use euclidean distance on lat/lon"]
    # Standard reading + revision -> noise, ignored.
    assert apply_interpretation_review(decisions, _probe(standard=True, revised_texts=revised)) == decisions
    # Non-standard reading + revision -> adopted wholesale.
    assert apply_interpretation_review(decisions, _probe(standard=False, revised_texts=revised)) == revised
    # Non-standard but empty revision -> keep original (fail-open).
    assert apply_interpretation_review(decisions, _probe(standard=False, revised_texts=[])) == decisions


# --- unit_graph wiring -------------------------------------------------------

def test_unit_graph_registers_candidates_and_injects_block(tmp_path, monkeypatch):
    """Two attempts with different outputs must produce a comparison block in
    the third attempt's prompts, and candidates must flow through state."""
    from acid_agent.graphs import unit_graph as ug
    from acid_agent.schemas import ExecResult
    from acid_agent.tracer import Tracer
    from acid_agent.workspace import Workspace

    prompts: list[str] = []
    calls = {"n": 0}

    def fake_ask(prompt):
        if "Condense" in prompt:
            return "obs"
        return "```python\nprint('done')\n```"

    def fake_run_claude(prompt, cwd=None):
        prompts.append(prompt)
        return ExecResult(ok=True, stdout="result: 295", stderr="", returncode=0)

    def fake_ask_structured(prompt, schema):
        calls["n"] += 1
        if schema.__name__ == "Decisions":
            return ug.Decisions(decisions=[])
        if schema.__name__ == "InterpretationReview":
            return InterpretationReview(chosen_reading_standard=True)
        from acid_agent.schemas import Reflection
        return Reflection(ok=True, feedback="")

    # Different stdout per attempt: 295 then 274, so a conflict exists by attempt 3.
    outputs = iter(["result: 295", "result: 274", "result: 274"])

    def fake_run_script(self, rel_path, timeout_s=180):
        if rel_path.startswith("explore"):
            return ExecResult(ok=True, stdout="col v rows 3", stderr="", returncode=0)
        return ExecResult(ok=True, stdout=next(outputs), stderr="", returncode=0)

    monkeypatch.setattr(ug, "ask", fake_ask)
    monkeypatch.setattr(ug, "run_claude", fake_run_claude)
    monkeypatch.setattr(ug, "ask_structured", fake_ask_structured)
    monkeypatch.setattr(ug, "validate_unit", lambda *a, **k: _failing_report())
    monkeypatch.setattr(type(Workspace.create(tmp_path, "t", None)), "run_script", fake_run_script)

    run_id = uuid.uuid4()
    ws = Workspace.create(tmp_path, "t", seed_files={"data.csv": "x\n1\n"})
    tracer = Tracer(run_id, logs_dir=str(tmp_path / "logs"))
    graph = ug.build_unit_graph(ws, tracer, run_id)
    state = graph.invoke({
        "task": "count", "unit_index": 0, "goal": "count cities",
        "exploration_budget": 1, "evidence_summary": "", "prior_observations": "",
        "decisions": [], "code": "", "exec_stdout": "", "exec_stderr": "",
        "attempt": 0, "feedback": "", "report": {}, "status": "running",
        "candidates": [],
    })
    tracer.close()

    assert state["candidates"] == [
        {"value": 295.0, "attempt": 1, "method": "count cities"},
        {"value": 274.0, "attempt": 2, "method": "count cities"},
        {"value": 274.0, "attempt": 3, "method": "count cities"},
    ]
    # Attempt 3's prompts (decisions + codegen) must carry the comparison block.
    assert any("ANSWER CANDIDATE COMPARISON" in p for p in prompts)


def _failing_report():
    from acid_agent.schemas import ValidationReport
    return ValidationReport(
        passed=False, review_decision="retry", components={},
        feedback="rejected", watchlist=[], probes={},
    )
