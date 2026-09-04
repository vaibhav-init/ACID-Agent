"""Regression tests for the graders and the KramaBench seed loader.

These pin the rules that previously produced fake zeros on correct answers:
scan-every-number grading, unicode-minus normalization, diacritic stripping,
partial list credit, ungradeable-type exclusion, and the 4-way seed-file
resolution (exact path -> glob -> directory recursion -> basename search).
"""

import pytest

from acid_agent.eval.kramabench import _name_grader, _num_grader
from acid_agent.eval.kramabench_tasks import (
    KramaTask,
    _norm,
    grade_exact,
    grade_numeric,
    grade_task,
    is_gradeable,
)


def _task(tmp_path, sources, answer=1.0, answer_type="numeric_exact"):
    return KramaTask(
        id="t", domain="d", question="q", answer=answer, answer_type=answer_type,
        data_sources=list(sources), data_dir=tmp_path,
    )


def _task_obj(answer, answer_type):
    return KramaTask(id="t", domain="d", question="q", answer=answer,
                     answer_type=answer_type, data_sources=[])


# --- numeric grading --------------------------------------------------

def test_numeric_scans_every_number():
    # The original bug: taking the first number scored this 0.0 against 5.28.
    assert grade_numeric(5.28, "Across all 99 rows the average is 5.28") == 1.0


def test_numeric_unicode_minus():
    # U+2212 is what models emit for negatives; without normalization the
    # value read as +0.008004 and a six-decimal-correct answer scored 0.0.
    assert grade_numeric(-0.008004, "correlation = \u22120.008004") == 1.0


def test_numeric_strips_commas():
    assert grade_numeric(1234.0, "total was 1,234 units") == 1.0


def test_numeric_tolerance_band():
    assert grade_numeric(100.0, "about 99.6") == 1.0
    assert grade_numeric(100.0, "about 99.0") == 0.0


def test_numeric_rejects_wrong_value():
    assert grade_numeric(3.14159, "the answer is 2.71828") == 0.0


def test_numeric_no_number_is_zero():
    assert grade_numeric(7.0, "I could not compute it") == 0.0


# --- string / list grading --------------------------------------------

def test_norm_strips_diacritics():
    assert _norm("São Paulo") == _norm("sao paulo")


def test_exact_string_matches_without_diacritics():
    assert grade_exact("São Paulo", "The answer is sao Paulo.") == 1.0


def test_exact_list_partial_credit():
    assert grade_exact(["a", "b", "c"], "a and b appear here") == pytest.approx(2 / 3)


def test_exact_numeric_delegates():
    assert grade_exact(42, "it is 42") == 1.0


# --- grade_task dispatch ----------------------------------------------

def test_ungradeable_type_returns_none():
    assert grade_task(_task_obj("prose", "string_approximate"), "anything") is None
    assert grade_task(_task_obj(["x"], "list_approximate"), "anything") is None
    assert not is_gradeable(_task_obj("prose", "string_approximate"))


def test_missing_answer_key_scores_zero():
    # Current behavior: a task with no answer key reads as wrong, NOT as
    # unjudgeable. Pinned so changing it is a deliberate decision.
    assert grade_task(_task_obj(None, "numeric_exact"), "42") == 0.0


def test_approx_numeric_uses_wider_tolerance():
    assert grade_task(_task_obj(100.0, "numeric_approximate"), "97") == 1.0
    assert grade_task(_task_obj(100.0, "numeric_exact"), "97") == 0.0


def test_exact_numeric_krama_tolerance():
    # Six-decimal correlations must survive the 0.005 relative band.
    assert grade_task(_task_obj(0.123456, "numeric_exact"), "0.12345") == 1.0


# --- synthetic graders -------------------------------------------------

def test_synthetic_num_grader_any_number():
    g = _num_grader(99.0)
    assert g("there are 99 rows, average 5.3") == 1.0
    assert g("nothing here") == 0.0


def test_synthetic_name_grader_case_insensitive():
    assert _name_grader("Rome")("the city is rome") == 1.0


# --- seed_files resolution --------------------------------------------

def _write(tmp_path, rel, data=b"data"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return data


def test_seed_exact_relative_path(tmp_path):
    data = _write(tmp_path, "alpha.csv", b"a,b\n1,2\n")
    assert _task(tmp_path, ["alpha.csv"]).seed_files == {"alpha.csv": data}


def test_seed_glob_pattern(tmp_path):
    d1 = _write(tmp_path, "beta/b1.csv", b"1")
    d2 = _write(tmp_path, "beta/b2.csv", b"2")
    assert _task(tmp_path, ["beta/*"]).seed_files == {"beta/b1.csv": d1, "beta/b2.csv": d2}


def test_seed_directory_recursion(tmp_path):
    d1 = _write(tmp_path, "beta/b1.csv", b"1")
    d2 = _write(tmp_path, "beta/sub/b2.csv", b"2")
    assert _task(tmp_path, ["beta"]).seed_files == {"beta/b1.csv": d1, "beta/sub/b2.csv": d2}


def test_seed_basename_finds_nested_file(tmp_path):
    # The legal/ failure mode: exact path matched nothing and every task
    # silently seeded an empty workspace.
    data = _write(tmp_path, "gamma/deep/delta.csv", b"d")
    assert _task(tmp_path, ["delta.csv"]).seed_files == {"delta.csv": data}


def test_seed_exact_wins_over_nested_copy(tmp_path):
    top = _write(tmp_path, "alpha.csv", b"top")
    _write(tmp_path, "gamma/deep/alpha.csv", b"nested")
    assert _task(tmp_path, ["alpha.csv"]).seed_files == {"alpha.csv": top}


def test_seed_preserves_binary_bytes(tmp_path):
    blob = bytes(range(256))
    _write(tmp_path, "input.xlsx", blob)
    assert _task(tmp_path, ["input.xlsx"]).seed_files["input.xlsx"] == blob
