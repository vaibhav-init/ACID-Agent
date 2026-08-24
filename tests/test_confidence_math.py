"""Pure-logic tests for the confidence engine (no model download needed)."""

from acid_agent.confidence import divergence, extract_code_spans

CODE = '''
import pandas as pd
df = pd.read_csv("orders.csv")
if df["region"].isna().any():
    df = df.dropna(subset=["region"])
north = df[df["region"] == "north"]
total = north.assign(rev=north.qty * north.unit_price).groupby("month").agg({"rev": "sum"})
print(total.corr())
'''


def test_divergence_relative():
    # relative form: |a-b| / max(a,b)
    assert divergence(0.5, 0.5) == 0.0
    assert abs(divergence(0.8, 0.3) - 0.625) < 1e-9
    assert abs(divergence(0.3, 0.8) - 0.625) < 1e-9  # symmetric


def test_extract_spans_finds_control_flow():
    spans = extract_code_spans(CODE)
    assert any("dropna" in s for s in spans), f"control-flow span missing: {spans}"


def test_extract_spans_finds_key_calls():
    spans = extract_code_spans(CODE)
    assert any("groupby" in s for s in spans)
    assert any("corr" in s for s in spans)


def test_extract_spans_handles_syntax_error():
    bad = "def broken(:" + chr(10) + "  pass"
    spans = extract_code_spans(bad)
    assert len(spans) == 1 and "broken" in spans[0]


def test_extract_spans_empty_code():
    assert extract_code_spans("") == []