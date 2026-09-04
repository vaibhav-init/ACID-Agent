"""Atomic Semantic Transaction Unit — LangGraph state machine.

explore (read-only, redundancy-bounded)
  -> extract_decisions
  -> generate_execute  (Claude Code writes & runs a unit script; we re-run it for a clean signal)
  -> validate          (execution + reflection + probability contrast + evidence surprise)
       pass -> commit   (git commit + memory evolution)
       fail -> rollback -> retry generate (<= MAX_RETRIES_PER_UNIT) or fail the unit

The workspace snapshot/rollback around retries is what makes failed attempts
leave ZERO trace in the workspace — semantic atomicity.
"""

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .. import confidence, memory, review
from ..config import get_conn, get_settings
from ..llm import ask, ask_structured
from ..claude_runner import run_claude
from ..schemas import Decision, Evidence, ExecResult
from ..validation import validate_unit
from ..workspace import Workspace


class Decisions(BaseModel):
    decisions: list[Decision] = []


class UnitState(TypedDict):
    task: str
    unit_index: int
    goal: str
    exploration_budget: int          # rounds allowed for this unit's exploration phase
    evidence_summary: str            # consolidated evidence (input + accumulated)
    prior_observations: str          # this unit's own explore-round summaries
    decisions: list[str]
    code: str
    exec_ok: bool                   # did the unit script run cleanly?
    exec_stdout: str
    exec_stderr: str
    exec_returncode: int            # real exit code; the gate classifies on it
    attempt: int
    feedback: str                    # validation feedback injected on retry
    report: dict
    status: str                      # running | committed | failed
    candidates: list[dict]           # per-attempt answer candidates (review.py)


EXPLORE_PROMPT = """You are exploring data READ-ONLY inside the current directory.

Unit goal: {goal}

Known evidence so far:
{evidence}

Long-term memory:
{memory}

Instructions:
- Inspect files/schemas/sample rows to make progress on the goal. DO NOT modify any file.
- Focus on what is NEW relative to the known evidence above.
- Finish by printing a short section 'OBSERVATIONS:' with concrete findings."""


def _summarize_observation(raw_output: str, goal: str) -> str:
    prompt = f"""Condense the following raw exploration output into 3-6 concrete factual
observations useful for the goal below. Facts only, no speculation.

Goal: {goal}
Raw output:
{raw_output[:6000]}"""
    return ask(prompt).strip()


def _extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if m:
        return max(m, key=len).strip()
    return ""


def build_unit_graph(ws: Workspace, tracer, run_id, task_slug: str | None = None):
    s = get_settings()

    # ---------- nodes ----------

    def explore(state: UnitState) -> dict:
        rounds = max(s.exploration_min_rounds, min(state["exploration_budget"], s.exploration_max_rounds))
        priors = state["prior_observations"]
        evidence = state["evidence_summary"]
        for rnd in range(rounds):
            prompt = EXPLORE_PROMPT.format(
                goal=state["goal"],
                evidence=evidence or "(none yet)",
                memory=memory.context_block(task_slug=task_slug) or "(empty)",
            )
            try:
                # Primary: backbone writes ONE read-only profiling snippet, executed directly.
                # ~5-12s vs ~30-90s for a full claude session; same evidence for the gate.
                code = ask(f"""Write ONE short read-only python script (pandas is available) that explores the
data files in the current directory for this goal and prints concrete findings.
It must only READ files. Return only one python code block.

{prompt}""")
                snippet = _extract_code(code) or "import os; print(os.listdir())"
                r = ws.run_code(snippet, name=f"explore_u{state['unit_index']}_r{rnd}.py")
                obs_raw = r.stdout + chr(10) + r.stderr
                if not r.ok and not r.stdout.strip():
                    # Fallback: scripted profiling failed — escalate to a full claude session.
                    res = run_claude(prompt, cwd=ws.root)
                    obs_raw = res.stdout if res.ok else res.stderr
                obs = _summarize_observation(obs_raw, state["goal"])
            except Exception as e:
                # Exploration is best-effort evidence gathering: a slow/failed
                # router round must not kill the transaction unit.
                tracer.log("explore_round_failed", unit_index=state["unit_index"], round=rnd, error=str(e)[:300])
                break
            tracer.log("explore", unit_index=state["unit_index"], round=rnd, observation=obs)

            # Redundancy check: PMI/token (nats) of the new observation against
            # the priors. High => already predicted by an earlier round => stop.
            if priors and rnd >= s.exploration_min_rounds - 1:
                try:
                    red = confidence.exploration_redundancy(obs, priors)
                    tracer.log("exploration_redundancy", value=red)
                    if red > s.redundancy_threshold:
                        break
                except Exception:
                    pass  # local model unavailable -> keep exploring within budget
            priors = (priors + chr(10) + obs).strip()
            evidence = (evidence + chr(10) + obs).strip()
        return {"prior_observations": priors, "evidence_summary": evidence}

    def extract_decisions(state: UnitState) -> dict:
        # On retries this node runs again WITH the gate feedback, so decisions can
        # be revised instead of staying frozen across attempts.
        feedback_note = (
            f"""

PREVIOUS ATTEMPT WAS REJECTED BY VALIDATION. Revise decisions accordingly: {state['feedback']}"""
            if state["feedback"]
            else ""
        )
        # Reactive review (port of the reference's AnswerCandidateTracker): when
        # two attempts produced different answers, make the model compare method
        # choices instead of picking on recency.
        cand_block = review.candidate_block(state.get("candidates") or []) if s.answer_candidates else None
        if cand_block:
            feedback_note += chr(10) + chr(10) + cand_block
        prompt = f"""From the evidence below, extract the 2-5 critical analytical DECISIONS
needed for this unit goal (joins, filters, granularity, formulas). Each must cite evidence.

Unit goal: {state['goal']}
Evidence:
{state['evidence_summary'] or '(none)'}{feedback_note}"""
        result = ask_structured(prompt, Decisions)
        decisions = [d.text for d in result.decisions]

        # Proactive interpretation review: challenge the FIRST reading of an
        # ambiguous goal before any code exists. First attempt only — retries
        # already carry gate feedback plus the candidate-comparison block.
        if s.interpretation_review and state.get("attempt", 0) == 0 and decisions:
            try:
                probe = ask_structured(
                    review.INTERPRETATION_PROMPT.format(
                        task=state["task"],
                        goal=state["goal"],
                        evidence=(state["evidence_summary"] or "(none)")[:1500],
                        decisions=chr(10).join("- " + d for d in decisions),
                    ),
                    review.InterpretationReview,
                )
                applied = review.apply_interpretation_review(decisions, probe)
                tracer.log(
                    "interpretation_review",
                    unit_index=state["unit_index"],
                    ambiguous_terms=probe.ambiguous_terms[:8],
                    standard_reading=probe.chosen_reading_standard,
                    revised=applied != decisions,
                )
                decisions = applied
            except Exception as e:
                # Fail-open: an unavailable backbone must not kill the unit.
                tracer.log("interpretation_review_failed", unit_index=state["unit_index"], error=str(e)[:300])

        tracer.log("decisions", unit_index=state["unit_index"], attempt=state["attempt"], decisions=decisions)
        return {"decisions": decisions}

    def generate_execute(state: UnitState) -> dict:
        attempt = state["attempt"] + 1
        candidates = list(state.get("candidates") or [])
        feedback_note = (
            f"""

PREVIOUS ATTEMPT WAS REJECTED. Fix this feedback: {state['feedback']}"""
            if state["feedback"]
            else ""
        )
        cand_block = review.candidate_block(candidates) if s.answer_candidates else None
        if cand_block:
            feedback_note += chr(10) + chr(10) + cand_block
        prompt = f"""In the current directory, write a python script named `unit{state['unit_index']}.py`
that accomplishes the unit goal using pandas (files are CSVs in '.'). Then RUN it and show its output.

Unit goal: {state['goal']}
Decisions to implement exactly:
{chr(10).join('- ' + d for d in state['decisions'])}
Known evidence:
{state['evidence_summary'][:2000]}{feedback_note}

The script must PRINT its key results clearly."""
        res = run_claude(prompt, cwd=ws.root)
        script_path = ws.root / f"unit{state['unit_index']}.py"
        code = script_path.read_text(encoding="utf-8") if script_path.exists() else ""

        if not code:
            # Fallback when claude returned nothing: backbone writes the script directly.
            try:
                code = _extract_code(ask(prompt)) or "# no code produced"
            except Exception as e:
                tracer.log("codegen_fallback_failed", unit_index=state["unit_index"], error=str(e)[:300])
                code = "# no code produced"
            script_path.write_text(code, encoding="utf-8")

        # Clean deterministic execution signal for the gate
        run: ExecResult = ws.run_script(f"unit{state['unit_index']}.py")
        # Register this attempt's answer candidate (reactive review); a future
        # retry then sees the disagreement block instead of silently switching.
        if s.answer_candidates:
            cand = review.last_number(run.stdout)
            if cand is not None:
                candidates.append({"value": cand, "attempt": attempt, "method": state["goal"][:120]})
        tracer.log(
            "generate_execute",
            unit_index=state["unit_index"],
            attempt=attempt,
            ok=run.ok,
            stdout=run.stdout[-1500:],
            stderr=run.stderr[-800:],
        )
        return {
            "attempt": attempt,
            "code": code,
            "exec_ok": run.ok,
            "exec_stdout": run.stdout,
            "exec_stderr": run.stderr,
            "exec_returncode": run.returncode,
            "candidates": candidates,
        }

    def validate(state: UnitState) -> dict:
        report = validate_unit(
            task=state["task"],
            evidence_summary=state["evidence_summary"],
            decisions=state["decisions"],
            code=state["code"],
            exec_result=ExecResult(
                ok=state.get("exec_ok", False),
                stdout=state["exec_stdout"],
                stderr=state["exec_stderr"],
                returncode=state.get("exec_returncode", 0),
            ),
            run_id=run_id,
            unit_index=state["unit_index"],
            attempt=state["attempt"],
            tracer=tracer,
            goal=state["goal"],
        )
        return {"report": report.model_dump(), "feedback": report.feedback}

    def commit(state: UnitState) -> dict:
        sha = ws.commit(f"unit-{state['unit_index']}: {state['goal'][:80]}")
        summary = f"{state['goal']} => {state['exec_stdout'][-400:].strip()}"
        try:
            memory.evolve_from_unit(summary, run_id=run_id, tracer=tracer, task_slug=task_slug)
        except Exception as e:
            tracer.log("memory_evolve_failed", error=str(e))
        tracer.log("commit", unit_index=state["unit_index"], git_commit=sha)
        with get_conn() as conn:
            conn.execute(
                """UPDATE units SET status='committed', attempts=%s, summary=%s, git_commit=%s
                   WHERE run_id=%s AND unit_index=%s""",
                (state["attempt"], summary[:500], sha, run_id, state["unit_index"]),
            )
        return {"status": "committed"}

    def fail_unit(state: UnitState) -> dict:
        ws.rollback()  # failed attempts leave zero trace
        tracer.log("unit_failed", unit_index=state["unit_index"])
        with get_conn() as conn:
            conn.execute(
                "UPDATE units SET status='failed', attempts=%s WHERE run_id=%s AND unit_index=%s",
                (state["attempt"], run_id, state["unit_index"]),
            )
        return {"status": "failed"}

    def retry_rollback(state: UnitState) -> dict:
        ws.rollback()  # erase rejected attempt before regenerating
        return {}

    # ---------- wiring ----------

    def after_validate(state: UnitState) -> str:
        report = state.get("report") or {}
        if report.get("passed"):
            return "commit"
        if state["attempt"] <= s.max_retries_per_unit:
            return "retry"
        return "fail"

    g = StateGraph(UnitState)
    g.add_node("explore", explore)
    g.add_node("extract_decisions", extract_decisions)
    g.add_node("generate_execute", generate_execute)
    g.add_node("validate", validate)
    g.add_node("commit", commit)
    g.add_node("fail_unit", fail_unit)
    g.add_node("retry_rollback", retry_rollback)

    g.add_edge(START, "explore")
    g.add_edge("explore", "extract_decisions")
    g.add_edge("generate_execute", "validate")
    g.add_conditional_edges(
        "validate",
        after_validate,
        {"commit": "commit", "retry": "retry_rollback", "fail": "fail_unit"},
    )
    g.add_edge("retry_rollback", "extract_decisions")  # retries may revise decisions using gate feedback
    g.add_edge("extract_decisions", "generate_execute")
    g.add_edge("commit", END)
    g.add_edge("fail_unit", END)

    return g.compile()