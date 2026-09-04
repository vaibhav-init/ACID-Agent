"""Validation gate (semantic consistency enforcement).

Four components are scored, each `pass` / `watch` / `retry` / `skipped`:

  1. execution_observation — structured exit code / stdout / stderr classification
  2. reasoning_observation — backbone LLM critique against the unit goal
  3. probability_contrast  — small-LM referee on evidence-vs-code conflicts the
                             backbone already flagged (no conflict => no signal)
  4. evidence_surprise     — small-LM signal over decision-relevant code spans;
                             `Settings.gate_semantics` picks the paper rule
                             (retry when divergence is LOW) or the reference
                             rule (retry when surprise is HIGH). Both metrics
                             are always computed and persisted.

`decision_surprise` is computed and persisted but **never gates**, matching the
reference's anchor_decision_surprise ("diagnostic-only: it never decides retry
policy by itself").

Only a `retry` component fails the attempt. `watch` items are recorded on the
report and let the attempt through, so a soft signal cannot burn the retry
budget by itself. A failed attempt is rolled back by the caller
(workspace.rollback()) and leaves zero trace.

Every confidence signal still fails **open**: if the local model is missing or a
probe cannot be built, that component is `skipped` rather than red.
"""

from typing import Literal

from pydantic import BaseModel

from . import confidence
from .config import get_settings
from .llm import ask_structured
from .schemas import ExecResult, ValidationReport
from .tracing import traced

STATUS_PASS = "pass"
STATUS_WATCH = "watch"
STATUS_RETRY = "retry"
STATUS_SKIPPED = "skipped"


class Reflection(BaseModel):
    ok: bool
    feedback: str


class AnchorCheck(BaseModel):
    """One evidence anchor compared against what the code actually implements."""

    anchor_id: str = ""
    decision_type: str = ""
    status: Literal["match", "conflict", "unclear"] = "unclear"
    current_policy: str = ""
    expected_policy: str = ""
    same_anchor_policy: bool = False
    comparison_valid: bool = False
    reason: str = ""


class AnchorAlignment(BaseModel):
    anchor_checks: list[AnchorCheck] = []


REFLECT_PROMPT = """You are a strict reviewer for a data-analysis agent.

Task:
{task}

Unit goal (review THIS unit only):
{goal}

Evidence gathered during exploration:
{evidence}

Decisions made:
{decisions}

Code that was executed:
{code}

Execution output:
{output}

Question: does this unit's work accomplish the UNIT GOAL, grounded in the evidence?
- The unit is one step of a larger task: do NOT demand the final answer here.
- Flag unsupported claims, decisions contradicting the evidence, ignored data quirks,
  or work that fails the unit goal.
- If the unit goal is reasonably accomplished, approve.
Reply with ok=true/false and one short actionable feedback sentence."""


ALIGNMENT_PROMPT = """Compare the EVIDENCE against what the CODE actually implements.

Task:
{task}

Evidence gathered during exploration:
{evidence}

Code that was executed:
{code}

For each concrete analytical decision point the evidence speaks to (join key, filter,
null handling, granularity, unit conversion, date parsing, aggregation, dedup...),
emit one check:
- anchor_id: short slug for the evidence point, e.g. 'region_has_nulls'
- decision_type: the kind of decision, e.g. 'null_handling'
- status: 'conflict' ONLY if the code contradicts the evidence; 'match' if it
  follows it; 'unclear' if the evidence does not settle it
- current_policy: what the CODE does, as a short noun phrase (e.g. 'drop rows with null region')
- expected_policy: what the EVIDENCE supports instead, same grammatical form
- same_anchor_policy: true only if both policies answer the SAME evidence point
- comparison_valid: true only if the two policies are genuine alternatives
- reason: one short sentence

Rules:
- current_policy and expected_policy must be directly comparable phrasings of the
  same decision. Never compare across different decision points.
- Report a conflict only when the evidence genuinely contradicts the code. Do not
  invent conflicts; an empty list is the correct answer for well-grounded code."""


def _classify_execution(exec_result: ExecResult) -> tuple[str, list[str]]:
    """Structured execution classification (reference: objective_confidence.py).

    Deliberately avoids traceback keyword sniffing — it reads exit code, stdout
    and stderr only.
    """
    stdout = (exec_result.stdout or "").strip()
    stderr = (exec_result.stderr or "").strip()

    if exec_result.returncode != 0 or not exec_result.ok:
        return "error", [f"execution exit_code={exec_result.returncode}"]
    if not stdout:
        note = f"; stderr output was present ({len(stderr)} chars)" if stderr else ""
        return "empty_observation", ["exit_code=0 but execution produced no stdout" + note]
    if stderr:
        return "stderr_nonempty", [f"exit_code=0 with stderr output ({len(stderr)} chars)"]
    return "standard_output", [f"exit_code=0 with stdout output ({len(stdout)} chars)"]


def _component(status: str, signals: list[str]) -> dict:
    return {"status": status, "signals": signals}


def _alignment_conflicts(task: str, evidence_summary: str, code: str, tracer=None) -> list[dict]:
    """Backbone pass that finds evidence-vs-code conflicts worth arbitrating.

    Gating the probe on this is the point of fix #3: the small model is a referee
    on a disagreement the strong model already identified, not a detector run over
    every span. No conflict here means probability_contrast is skipped entirely.
    """
    conflicts: list[dict] = []
    try:
        result: AnchorAlignment = ask_structured(
            ALIGNMENT_PROMPT.format(
                task=task,
                evidence=evidence_summary[:2500],
                code=code[:4000],
            ),
            AnchorAlignment,
        )
        checks = result.anchor_checks
        for c in checks:
            if c.status != "conflict":
                continue
            # The reference refuses to score a pair unless the LLM certifies both
            # policies answer the same anchor; a cross-dimensional comparison makes
            # the likelihood ratio meaningless.
            if not (c.same_anchor_policy and c.comparison_valid):
                continue
            if not c.current_policy.strip() or not c.expected_policy.strip():
                continue
            conflicts.append(c.model_dump())
    except Exception as e:
        # A malformed structured response must not crash the gate; no conflict
        # simply means probability_contrast is skipped for this attempt.
        if tracer:
            tracer.log("anchor_alignment_failed", error=str(e)[:300])
        return []
    if tracer:
        tracer.log("anchor_alignment", n_checks=len(checks), n_conflicts=len(conflicts))
    return conflicts


@traced("validation_gate", drop=("tracer",))
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
    goal: str = "",
) -> ValidationReport:
    s = get_settings()
    report = ValidationReport()
    components: dict[str, dict] = {}

    # --- Component 1: execution observation -----------------------------
    obs_class, exec_signals = _classify_execution(exec_result)
    report.execution_ok = obs_class in ("standard_output", "stderr_nonempty")
    if obs_class in ("error", "empty_observation"):
        components["execution_observation"] = _component(STATUS_RETRY, exec_signals)
    elif obs_class == "stderr_nonempty":
        components["execution_observation"] = _component(STATUS_WATCH, exec_signals)
    else:
        components["execution_observation"] = _component(STATUS_PASS, exec_signals)

    # --- Component 2: reasoning observation (LLM reflection) -------------
    try:
        refl: Reflection = ask_structured(
            REFLECT_PROMPT.format(
                task=task,
                goal=goal or "(see task)",
                evidence=evidence_summary[:2500],
                decisions=chr(10).join(f"- {d}" for d in decisions) or "(none)",
                code=code[:4000],
                output=(exec_result.stdout + chr(10) + exec_result.stderr)[:3000],
            ),
            Reflection,
        )
        report.reflection_ok = refl.ok
        report.feedback = refl.feedback
        components["reasoning_observation"] = _component(
            STATUS_PASS if refl.ok else STATUS_RETRY,
            [refl.feedback[:300]] if refl.feedback else [],
        )
    except Exception as e:
        report.reflection_ok = True  # don't block on judge failure
        components["reasoning_observation"] = _component(
            STATUS_SKIPPED, [f"reflection unavailable: {e}"[:300]]
        )
        if tracer:
            tracer.log("reflection_skipped", error=str(e))

    # --- Component 3: probability contrast (LLM-flagged conflicts only) --
    conflicts = _alignment_conflicts(task, evidence_summary, code, tracer=tracer)
    try:
        contrast = confidence.probability_contrast(conflicts, task, evidence_summary)
    except Exception as e:
        contrast = {"available": False, "reason": f"scorer unavailable: {e}"[:200]}
        if tracer:
            tracer.log("contrast_skipped", error=str(e))
    report.probes["probability_contrast"] = contrast
    if contrast.get("available"):
        report.contrast_min_ratio = contrast.get("min_ratio")
        statuses = {p["status"] for p in contrast.get("probes", [])}
        status = (
            STATUS_RETRY if STATUS_RETRY in statuses
            else STATUS_WATCH if STATUS_WATCH in statuses
            else STATUS_PASS
        )
        components["probability_contrast"] = _component(
            status,
            [
                f"{p['decision_type'] or p['anchor_id']}: current={p['current']!r} "
                f"alternative={p['alternative']!r} ratio={p['ratio']:.4f}"
                for p in contrast["probes"]
                if p["status"] != STATUS_PASS
            ][:4],
        )
    else:
        components["probability_contrast"] = _component(
            STATUS_SKIPPED, [contrast.get("reason", "no signal")]
        )

    # --- Component 4: evidence surprise over code spans ------------------
    try:
        span = confidence.code_span_surprise(code, evidence_summary, task)
    except Exception as e:
        span = {"available": False, "reason": f"scorer unavailable: {e}"[:200]}
        if tracer:
            tracer.log("confidence_skipped", error=str(e))
    report.probes["evidence_surprise"] = span
    if span.get("available"):
        # Both metrics are recorded on every attempt regardless of which one
        # gates, so one run supports the paper-vs-reference comparison.
        report.max_span_surprise = span.get("max_surprise")
        report.max_span_divergence = span.get("max_divergence")
        if s.gate_semantics == "reference":
            # High surprise => evidence suppresses the span => code contradicts it.
            worst = max(span["spans"], key=lambda sp: sp["surprise"])
            value, metric = report.max_span_surprise, "surprise"
            if value > s.span_surprise_retry:
                status = STATUS_RETRY
            elif value > s.span_surprise_warn:
                status = STATUS_WATCH
            else:
                status = STATUS_PASS
        else:
            # Paper: low divergence => evidence changed nothing => ungrounded.
            # Binary here: the paper states no watch band for this signal.
            worst = min(span["spans"], key=lambda sp: sp["divergence"])
            value, metric = report.max_span_divergence, "divergence"
            status = STATUS_RETRY if value < s.span_divergence_min else STATUS_PASS
        components["evidence_surprise"] = _component(
            status,
            [
                f"line {worst['line_no']} ({worst['category']}) {metric}="
                f"{worst[metric]:.4f} (max {value:.4f}, {s.gate_semantics} rule): "
                f"{worst['span'][:140]}"
            ] if status != STATUS_PASS else [],
        )
    else:
        components["evidence_surprise"] = _component(
            STATUS_SKIPPED, [span.get("reason", "no signal")]
        )

    # --- Diagnostic only: decision surprise (never gates) ----------------
    try:
        dec = confidence.decision_surprise(decisions, evidence_summary, task)
    except Exception as e:
        dec = {"available": False, "reason": f"scorer unavailable: {e}"[:200]}
    report.probes["decision_surprise"] = dec
    if dec.get("available"):
        report.decision_surprise = dec.get("max_surprise")

    # --- Merge: only a RETRY component fails the attempt -----------------
    report.components = components
    need_retry = any(c["status"] == STATUS_RETRY for c in components.values())
    report.watchlist = [
        f"{name}: {'; '.join(c['signals']) or c['status']}"
        for name, c in components.items()
        if c["status"] == STATUS_WATCH
    ]
    report.review_decision = (
        STATUS_RETRY if need_retry else (STATUS_WATCH if report.watchlist else STATUS_PASS)
    )
    passed = not need_retry
    report.passed = passed

    if s.gate_bypass and not passed:
        # Ablation arm: keep the real signals in the report/DB for analysis, but
        # let the attempt through so no retry or rollback is triggered.
        report.passed = True
        report.feedback = ""
        if tracer:
            tracer.log(
                "gate_bypassed",
                unit_index=unit_index,
                attempt=attempt,
                would_have_failed=True,
                review_decision=report.review_decision,
                max_span_surprise=report.max_span_surprise,
                max_span_divergence=report.max_span_divergence,
                contrast_min_ratio=report.contrast_min_ratio,
                decision_surprise=report.decision_surprise,
                execution_ok=report.execution_ok,
                reflection_ok=report.reflection_ok,
            )
        passed = True

    if not passed and not report.feedback:
        parts = []
        for name, c in components.items():
            if c["status"] != STATUS_RETRY:
                continue
            if name == "execution_observation":
                parts.append("fix the execution errors and print the key results")
            elif name == "probability_contrast":
                parts.append(
                    "the evidence supports a different choice than the code makes: "
                    + (c["signals"][0] if c["signals"] else "revisit the flagged decision")
                )
            elif name == "evidence_surprise":
                lead = (
                    "a key code step contradicts the explored evidence: "
                    if s.gate_semantics == "reference"
                    else "key code steps are not grounded in the explored evidence: "
                )
                parts.append(lead + (c["signals"][0] if c["signals"] else "re-ground the analysis"))
        report.feedback = "; ".join(parts)

    # Persist gate result
    if tracer:
        tracer.log(
            "validate",
            unit_index=unit_index,
            attempt=attempt,
            passed=passed,
            review_decision=report.review_decision,
            gate_semantics=s.gate_semantics,
            max_span_surprise=report.max_span_surprise,
            max_span_divergence=report.max_span_divergence,
            contrast_min_ratio=report.contrast_min_ratio,
            decision_surprise=report.decision_surprise,
            execution_ok=report.execution_ok,
            reflection_ok=report.reflection_ok,
            watchlist=report.watchlist,
            feedback=report.feedback,
        )
    try:
        import json

        from .config import get_conn

        with get_conn() as conn:
            conn.execute(
                """INSERT INTO validations (run_id, unit_index, attempt, decision_surprise,
                       max_span_surprise, max_span_divergence, gate_semantics,
                       contrast_min_ratio, review_decision, watchlist,
                       execution_ok, reflection_ok, passed, feedback)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    run_id, unit_index, attempt, report.decision_surprise,
                    report.max_span_surprise, report.max_span_divergence,
                    s.gate_semantics, report.contrast_min_ratio,
                    report.review_decision, json.dumps(report.watchlist),
                    report.execution_ok, report.reflection_ok, passed, report.feedback,
                ),
            )
    except Exception:
        pass

    return report
