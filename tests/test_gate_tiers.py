"""Gate merge semantics: pass / watch / retry, and the direction of each signal.

The old gate was inverted — it required HIGH divergence to PASS. These tests pin
the corrected direction, because a sign flip is invisible to any test that only
checks that a number was produced.
"""

import pytest

import acid_agent.confidence as conf
import acid_agent.validation as val
from acid_agent.schemas import ExecResult


def _probe_span(surprise: float, divergence: float = 1.0) -> dict:
    """Both metrics are carried on every span; the gate picks one."""
    return {
        "available": True, "max_surprise": surprise, "max_divergence": divergence,
        "n_scored": 1,
        "spans": [{"line_no": 1, "category": "call:merge", "span": "df.merge(o)",
                   "cond_logp": -2.0, "prior_logp": -2.0 + surprise,
                   "delta": -surprise, "surprise": surprise,
                   "divergence": divergence}],
    }


def _probe_contrast(ratio: float | None) -> dict:
    if ratio is None:
        return {"available": False, "min_ratio": None, "probes": [], "n_scored": 0,
                "reason": "no LLM-flagged conflict"}
    status = ("retry" if ratio < 0.25 else "watch" if ratio < 0.75 else "pass")
    return {
        "available": True, "min_ratio": ratio, "n_scored": 1,
        "probes": [{"anchor_id": "a", "decision_type": "join", "current": "x",
                    "alternative": "y", "current_logp": -1.0, "alternative_logp": -1.0,
                    "delta": 0.0, "ratio": ratio, "status": status}],
    }


@pytest.fixture()
def gate(monkeypatch):
    """validate_unit with the scorer, the LLM and Postgres all stubbed out."""
    monkeypatch.setattr(conf, "decision_surprise",
                        lambda *a, **k: {"available": True, "max_surprise": 0.0,
                                         "n_scored": 1, "decisions": []})
    monkeypatch.setattr(val, "get_conn", lambda: (_ for _ in ()).throw(RuntimeError("no db")),
                        raising=False)

    # Default to the reference rule for the pre-existing surprise tests; the
    # paper-rule tests below flip it explicitly.
    monkeypatch.setattr(val.get_settings(), "gate_semantics", "reference")

    def _run(span_surprise=0.0, span_divergence=1.0, contrast_ratio=None,
             reflection_ok=True, exec_result=None, decisions=("d",)):
        monkeypatch.setattr(conf, "code_span_surprise",
                            lambda *a, **k: _probe_span(span_surprise, span_divergence))
        monkeypatch.setattr(conf, "probability_contrast",
                            lambda *a, **k: _probe_contrast(contrast_ratio))
        monkeypatch.setattr(
            val, "ask_structured",
            lambda p, s: (val.AnchorAlignment(anchor_checks=[])
                          if s.__name__ == "AnchorAlignment"
                          else val.Reflection(ok=reflection_ok, feedback="" if reflection_ok else "bad")),
        )
        return val.validate_unit(
            task="t", evidence_summary="e", decisions=list(decisions), code="df.merge(o)",
            exec_result=exec_result or ExecResult(ok=True, stdout="R: 1", stderr="", returncode=0),
            run_id=None, unit_index=0, attempt=1, goal="g",
        )

    return _run


# ----------------------------------------------------------- span surprise

def test_low_span_surprise_passes(gate):
    # Evidence supports the code. Under the OLD gate this was a rejection.
    r = gate(span_surprise=0.02)
    assert r.passed and r.review_decision == "pass"


def test_mid_span_surprise_watches_but_does_not_retry(gate):
    r = gate(span_surprise=0.30)   # > warn 0.10, < retry 0.50
    assert r.passed, "a watch-level signal must not fail the attempt"
    assert r.review_decision == "watch"
    assert any("evidence_surprise" in w for w in r.watchlist)


def test_high_span_surprise_retries(gate):
    r = gate(span_surprise=0.80)   # > retry 0.50
    assert not r.passed and r.review_decision == "retry"
    assert r.components["evidence_surprise"]["status"] == "retry"


# ------------------------------------------------------ probability contrast

def test_no_flagged_conflict_skips_contrast(gate):
    # The whole point of fix #3: no LLM-flagged conflict => no probe => no signal.
    r = gate(contrast_ratio=None)
    assert r.components["probability_contrast"]["status"] == "skipped"
    assert r.passed
    assert r.contrast_min_ratio is None


def test_low_contrast_ratio_retries(gate):
    # Current policy 10x less likely than the evidence-backed alternative.
    r = gate(contrast_ratio=0.10)
    assert not r.passed and r.components["probability_contrast"]["status"] == "retry"


def test_mid_contrast_ratio_watches(gate):
    r = gate(contrast_ratio=0.50)
    assert r.passed and r.review_decision == "watch"


def test_high_contrast_ratio_passes(gate):
    r = gate(contrast_ratio=1.50)
    assert r.passed and r.review_decision == "pass"


# ---------------------------------------------------------------- execution

def test_nonzero_exit_retries(gate):
    r = gate(exec_result=ExecResult(ok=False, stdout="", stderr="boom", returncode=1))
    assert not r.passed
    assert r.components["execution_observation"]["status"] == "retry"


def test_empty_stdout_retries(gate):
    # exit 0 but nothing printed: the reference treats this as no observation.
    r = gate(exec_result=ExecResult(ok=True, stdout="", stderr="", returncode=0))
    assert not r.passed
    assert r.components["execution_observation"]["status"] == "retry"


def test_stderr_with_output_only_watches(gate):
    r = gate(exec_result=ExecResult(ok=True, stdout="R: 1", stderr="FutureWarning", returncode=0))
    assert r.passed and r.review_decision == "watch"


# --------------------------------------------------- decision surprise is diagnostic

def test_decision_surprise_never_gates(gate, monkeypatch):
    monkeypatch.setattr(conf, "decision_surprise",
                        lambda *a, **k: {"available": True, "max_surprise": 9.9,
                                         "n_scored": 1, "decisions": []})
    r = gate(span_surprise=0.0)
    assert r.decision_surprise == 9.9, "still recorded"
    assert r.passed, "diagnostic-only: it must never fail an attempt"
    assert "decision_surprise" not in r.components


# ------------------------------------------------------------- fail-open

def test_scorer_failure_skips_rather_than_blocks(monkeypatch):
    # Standalone, not via the `gate` fixture: that fixture re-patches the probes
    # inside _run and would overwrite these.
    def boom(*a, **k):
        raise RuntimeError("torch missing")

    monkeypatch.setattr(conf, "code_span_surprise", boom)
    monkeypatch.setattr(conf, "probability_contrast", boom)
    monkeypatch.setattr(conf, "decision_surprise", boom)
    monkeypatch.setattr(
        val, "ask_structured",
        lambda p, s: (val.AnchorAlignment(anchor_checks=[])
                      if s.__name__ == "AnchorAlignment"
                      else val.Reflection(ok=True, feedback="")),
    )
    r = val.validate_unit(
        task="t", evidence_summary="e", decisions=["d"], code="df.merge(o)",
        exec_result=ExecResult(ok=True, stdout="R", stderr="", returncode=0),
        run_id=None, unit_index=0, attempt=1, goal="g",
    )
    assert r.passed, "confidence signals must fail open"
    assert r.components["evidence_surprise"]["status"] == "skipped"
    assert r.components["probability_contrast"]["status"] == "skipped"
    assert r.max_span_surprise is None


def test_reflection_failure_skips_rather_than_blocks(monkeypatch):
    monkeypatch.setattr(conf, "code_span_surprise", lambda *a, **k: _probe_span(0.0))
    monkeypatch.setattr(conf, "probability_contrast", lambda *a, **k: _probe_contrast(None))
    monkeypatch.setattr(conf, "decision_surprise",
                        lambda *a, **k: {"available": False, "max_surprise": 0.0})

    def boom(p, s):
        raise RuntimeError("cli down")

    monkeypatch.setattr(val, "ask_structured", boom)
    r = val.validate_unit(
        task="t", evidence_summary="e", decisions=["d"], code="df.merge(o)",
        exec_result=ExecResult(ok=True, stdout="R", stderr="", returncode=0),
        run_id=None, unit_index=0, attempt=1, goal="g",
    )
    assert r.passed
    assert r.components["reasoning_observation"]["status"] == "skipped"


def test_reflection_rejection_still_retries(gate):
    r = gate(reflection_ok=False)
    assert not r.passed and r.components["reasoning_observation"]["status"] == "retry"


# ------------------------------------------------- paper vs reference semantics

def test_paper_rule_retries_on_LOW_divergence(gate, monkeypatch):
    # Paper §2.2.2: retry when the max code-span divergence is BELOW 0.50 --
    # the evidence changed nothing, so the code is not grounded in it.
    monkeypatch.setattr(val.get_settings(), "gate_semantics", "paper")
    r = gate(span_divergence=0.10, span_surprise=0.0)
    assert not r.passed and r.components["evidence_surprise"]["status"] == "retry"


def test_paper_rule_passes_on_HIGH_divergence(gate, monkeypatch):
    monkeypatch.setattr(val.get_settings(), "gate_semantics", "paper")
    r = gate(span_divergence=0.90, span_surprise=0.0)
    assert r.passed and r.components["evidence_surprise"]["status"] == "pass"


def test_the_two_rules_disagree_on_the_same_numbers(gate, monkeypatch):
    # The whole reason gate_semantics exists: one span, one pair of log-probs,
    # opposite verdicts. High surprise AND high divergence -> the reference rule
    # rejects (evidence contradicts) while the paper rule accepts (evidence
    # clearly mattered).
    monkeypatch.setattr(val.get_settings(), "gate_semantics", "reference")
    ref = gate(span_surprise=0.80, span_divergence=0.90)
    monkeypatch.setattr(val.get_settings(), "gate_semantics", "paper")
    paper = gate(span_surprise=0.80, span_divergence=0.90)
    assert not ref.passed and paper.passed


def test_both_metrics_recorded_whichever_rule_gates(gate, monkeypatch):
    monkeypatch.setattr(val.get_settings(), "gate_semantics", "paper")
    r = gate(span_surprise=0.42, span_divergence=0.77)
    assert r.max_span_surprise == 0.42 and r.max_span_divergence == 0.77


def test_paper_rule_has_no_watch_band_for_spans(gate, monkeypatch):
    # The paper states a single threshold for this signal, so it is binary.
    # Pin the threshold too: Settings is a cached singleton fed by .env, and
    # .env now carries a calibrated SPAN_DIVERGENCE_MIN (0.25).
    s = val.get_settings()
    monkeypatch.setattr(s, "gate_semantics", "paper")
    monkeypatch.setattr(s, "span_divergence_min", 0.50)
    assert gate(span_divergence=0.51).review_decision == "pass"
    assert gate(span_divergence=0.49).review_decision == "retry"
