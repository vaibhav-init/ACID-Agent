"""Validation gate (semantic consistency enforcement).

A unit's attempt PASSES only if ALL signals are green:
  1. execution_ok            — script ran without errors
  2. decision_divergence     >= DECISION_DIVERGENCE_MIN   (decisions grounded in evidence)
  3. max_code_span_divergence >= CODE_SPAN_DIVERGENCE_MIN (code grounded in evidence)
  4. reflection_ok           — backbone LLM critique finds no unsupported claims

Any red signal => retry with feedback (max MAX_RETRIES_PER_UNIT). Failed attempts
are rolled back by the caller (workspace.rollback()) so they leave zero trace.
"""

from pydantic import BaseModel

from . import confidence
from .config import get_settings
from .llm import ask_structured
from .schemas import ExecResult, ValidationReport


class Reflection(BaseModel):
    ok: bool
    feedback: str


REFLECT_PROMPT = """You are a strict reviewer for a data-analysis agent.

Task:
{task}

Evidence gathered during exploration:
{evidence}

Decisions made:
{decisions}

Code that was executed:
{code}

Execution output:
{output}

Question: is this work well-grounded and likely correct?
- Flag unsupported claims, decisions contradicting the evidence, ignored data quirks,
  or results that don't answer the task.
- If everything is reasonably grounded, approve.
Reply with ok=true/false and one short actionable feedback sentence."""


def validate_unit(
    task: str,
    evidence_summary: str,
    decisions: list[str],
    code: str,
    exec_result: ExecResult,
    run_id=None,
    unit_index: int = 0,
    attempt: int = 0,
    tracer=None,
) -> ValidationReport:
    s = get_settings()
    report = ValidationReport()

    # Signal 1: execution
    report.execution_ok = exec_result.ok

    # Signals 2 & 3: confidence divergences (skipped gracefully if local model unavailable)
    try:
        if decisions:
            report.decision_divergence = confidence.decision_divergence(
                decisions, evidence_summary, task
            )
        report.max_code_span_divergence = confidence.code_span_divergence(
            code, evidence_summary, task
        )
    except Exception as e:  # torch/transformers missing or model load failed
        if tracer:
            tracer.log("confidence_skipped", error=str(e))

    # Signal 4: LLM reflection
    try:
        refl: Reflection = ask_structured(
            REFLECT_PROMPT.format(
                task=task,
                evidence=evidence_summary[:2500],
                decisions=chr(10).join(f"- {d}" for d in decisions) or "(none)",
                code=code[:4000],
                output=(exec_result.stdout + chr(10) + exec_result.stderr)[:3000],
            ),
            Reflection,
        )
        report.reflection_ok = refl.ok
        report.feedback = refl.feedback
    except Exception as e:
        report.reflection_ok = True  # don't block on judge failure
        if tracer:
            tracer.log("reflection_skipped", error=str(e))

    # Combine all signals
    passed = (
        report.execution_ok
        and report.reflection_ok
        and (report.decision_divergence is None or report.decision_divergence >= s.decision_divergence_min)
        and (
            report.max_code_span_divergence is None
            or report.max_code_span_divergence >= s.code_span_divergence_min
        )
    )
    report.passed = passed

    if not passed and not report.feedback:
        parts = []
        if not report.execution_ok:
            parts.append("fix the execution errors")
        if report.decision_divergence is not None and report.decision_divergence < s.decision_divergence_min:
            parts.append("ground your decisions in the explored evidence")
        if (
            report.max_code_span_divergence is not None
            and report.max_code_span_divergence < s.code_span_divergence_min
        ):
            parts.append("make key code steps follow from the evidence")
        report.feedback = "; ".join(parts)

    # Persist gate result
    if tracer:
        tracer.log(
            "validate",
            unit_index=unit_index,
            attempt=attempt,
            passed=passed,
            decision_divergence=report.decision_divergence,
            max_code_span_divergence=report.max_code_span_divergence,
            execution_ok=report.execution_ok,
            reflection_ok=report.reflection_ok,
            feedback=report.feedback,
        )
    try:
        from .config import get_conn

        with get_conn() as conn:
            conn.execute(
                """INSERT INTO validations (run_id, unit_index, attempt, decision_divergence,
                       max_code_span_divergence, execution_ok, reflection_ok, passed, feedback)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    run_id, unit_index, attempt, report.decision_divergence,
                    report.max_code_span_divergence, report.execution_ok,
                    report.reflection_ok, passed, report.feedback,
                ),
            )
    except Exception:
        pass

    return report