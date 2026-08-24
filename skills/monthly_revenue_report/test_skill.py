"""Validation suite for the monthly_revenue_report skill.

Must pass before the skill is trusted by the router (paper: test-driven skill hub).
"""

import importlib.util
from pathlib import Path

import pandas as pd

SKILL_DIR = Path(__file__).parent


def _load_skill():
    spec = importlib.util.spec_from_file_location("skill", SKILL_DIR / "skill.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_csv(tmp_path, rows):
    p = tmp_path / "orders.csv"
    lines = ["date,revenue"] + [f"{d},{v}" for d, v in rows]
    p.write_text(chr(10).join(lines))
    return str(p)


def test_handles_mixed_date_formats(tmp_path):
    skill = _load_skill()
    # 2024-01 total = 100 + 50 ; 2024-02 total = 200 ; one row uses '/' format
    csv = _make_csv(
        tmp_path,
        [("2024-01-05", 100), ("2024/01/20", 50), ("2024-02-10", 200)],
    )
    out = skill.monthly_revenue(csv)
    assert abs(out["2024-01"] - 150.0) < 1e-6
    assert abs(out["2024-02"] - 200.0) < 1e-6


def test_reports_dropped_rows(tmp_path, capsys):
    skill = _load_skill()
    csv = _make_csv(tmp_path, [("2024-01-05", 100), ("not-a-date", 999), ("2024-01-06", "")])
    out = skill.monthly_revenue(csv)
    captured = capsys.readouterr().out
    assert "rows_dropped=2" in captured
    assert abs(out["2024-01"] - 100.0) < 1e-6


def test_output_is_pandas_series(tmp_path):
    skill = _load_skill()
    csv = _make_csv(tmp_path, [("2024-03-01", 42)])
    out = skill.monthly_revenue(csv)
    assert isinstance(out, pd.Series)