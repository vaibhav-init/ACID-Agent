"""Pure-logic tests for the confidence engine (no model download needed)."""

import pytest

from acid_agent.confidence import (
    _contrast_status,
    _support_status,
    extract_code_span_objects,
    extract_code_spans,
    surprise,
)

CODE = '''
import pandas as pd
df = pd.read_csv("orders.csv")
if df["region"].isna().any():
    df = df.dropna(subset=["region"])
north = df[df["region"] == "north"]
total = north.assign(rev=north.qty * north.unit_price).groupby("month").agg({"rev": "sum"})
print(total.corr())
'''


def test_surprise_is_one_sided():
    # Evidence SUPPRESSING the span (cond < prior) is the defect signal.
    assert surprise(-3.0, -2.0) == pytest.approx(1.0)
    # Evidence SUPPORTING the span is not a defect: clamps to zero rather than
    # registering as a divergence the way the old abs() form did.
    assert surprise(-2.0, -3.0) == 0.0
    assert surprise(-2.0, -2.0) == 0.0


def test_surprise_is_in_nats_not_normalized():
    # Raw delta, no division by anything — a 0.5 nat gap reads as 0.5.
    assert surprise(-2.5, -2.0) == pytest.approx(0.5)
    assert surprise(-20.5, -20.0) == pytest.approx(0.5)


def test_surprise_handles_unscorable():
    assert surprise(float("-inf"), -2.0) == 0.0
    assert surprise(-2.0, float("-inf")) == 0.0


def test_contrast_status_thresholds():
    # ratio = P(current)/P(evidence-backed alternative); LOW is bad.
    assert _contrast_status(0.10) == "retry"
    assert _contrast_status(0.50) == "watch"
    assert _contrast_status(1.20) == "pass"


def test_decision_support_status_is_diagnostic_scale():
    assert _support_status(0.90) == "unsupported"
    assert _support_status(0.30) == "weak_support"
    assert _support_status(0.05) == "supported"


def test_extract_spans_finds_control_flow():
    spans = extract_code_spans(CODE)
    assert any("dropna" in s for s in spans), f"control-flow span missing: {spans}"


def test_extract_spans_finds_key_calls():
    spans = extract_code_spans(CODE)
    assert any("groupby" in s for s in spans)
    assert any("read_csv" in s for s in spans)


def test_extract_spans_covers_calls_outside_any_allow_list():
    # The old extractor matched 10 hand-picked pandas method names and found
    # ZERO spans here — including the date parse the mixed-format trap turns on.
    spans = extract_code_spans(
        "df = pd.read_csv('a.csv')" + chr(10)
        + "df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')"
    )
    assert any("to_datetime" in s for s in spans)


def test_extract_spans_skips_non_decisions():
    # Imports, prints and empty-container init carry no analytical decision.
    code = (
        "import pandas as pd" + chr(10)
        + "rows = []" + chr(10)
        + "print('hello')" + chr(10)
        + "df.head()"
    )
    assert extract_code_spans(code) == []


def test_extract_spans_categorises():
    spans = extract_code_span_objects(CODE)
    cats = {sp.category for sp in spans}
    assert "control_flow" in cats
    assert "conditional_selection" in cats


def test_extract_spans_handles_syntax_error():
    bad = "def broken(:" + chr(10) + "  pass"
    spans = extract_code_spans(bad)
    assert len(spans) == 1 and "broken" in spans[0]


def test_extract_spans_empty_code():
    assert extract_code_spans("") == []


def test_spans_carry_preceding_code_as_prefix():
    # Each span is scored as a continuation of the code that leads up to it, so
    # the prefix must contain the earlier lines and not the span itself.
    spans = extract_code_span_objects(CODE)
    groupby = next(sp for sp in spans if "groupby" in sp.text)
    assert "read_csv" in groupby.code_prefix
    assert "groupby" not in groupby.code_prefix
    assert groupby.line_no > 1


def test_spans_are_ordered_by_line():
    spans = extract_code_span_objects(CODE)
    assert [sp.line_no for sp in spans] == sorted(sp.line_no for sp in spans)
