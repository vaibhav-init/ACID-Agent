"""End-to-end wiring test of the transaction-unit graph with fake LLM/OpenCode/confidence.

Requires Postgres running (docker compose up -d); skips otherwise.
Proves: commit path commits + rolls back nothing; failure path exhausts retries,
rolls back, and leaves zero trace.
"""

import uuid

import pytest

from acid_agent.config import get_conn


def _db_up() -> bool:
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running (docker compose up -d)")


@pytest.fixture()
def fake_env(monkeypatch):
    from pydantic import BaseModel

    import acid_agent.confidence as conf_mod
    import acid_agent.validation as val_mod
    from acid_agent.graphs import unit_graph
    from acid_agent.schemas import Decision
    from acid_agent.graphs.unit_graph import Decisions

    class Reflection(BaseModel):
        ok: bool = True
        feedback: str = ""

    def fake_run_opencode(prompt, cwd, timeout_s=None):
        class R:
            ok = True
            stdout = "OBSERVATIONS: data.csv has 3 rows, one column v."
            stderr = ""
            returncode = 0

        return R()

    def fake_ask(prompt):
        if "Condense" in prompt:
            return "data.csv has 3 rows with column v"
        # code generation fallback -> script that prints a result
        return (
            "```python"
            + chr(10)
            + "print('RESULT:', sum([1,2,3]))"
            + chr(10)
            + "```"
        )

    def fake_ask_structured(prompt, schema):
        if schema.__name__ == "Decisions":
            return Decisions(decisions=[Decision(id="d1", text="sum column v", rationale="evidence")])
        return Reflection(ok=True, feedback="")

    monkeypatch.setattr(unit_graph, "run_opencode", fake_run_opencode)
    monkeypatch.setattr(unit_graph, "ask", fake_ask)
    monkeypatch.setattr(unit_graph, "ask_structured", fake_ask_structured)
    monkeypatch.setattr(val_mod, "ask_structured", fake_ask_structured)
    monkeypatch.setattr(conf_mod, "decision_divergence", lambda *a, **k: 0.9)
    monkeypatch.setattr(conf_mod, "code_span_divergence", lambda *a, **k: 0.9)
    monkeypatch.setattr(unit_graph.memory, "evolve_from_unit", lambda *a, **k: [])
    return unit_graph


def _run_unit(fake_env, tmp_path, seed_code):
    from acid_agent.graphs.unit_graph import build_unit_graph
    from acid_agent.tracer import Tracer
    from acid_agent.workspace import Workspace

    run_id = uuid.uuid4()
    ws = Workspace.create(tmp_path, "u", seed_files={"data.csv": seed_code})
    tracer = Tracer(run_id, logs_dir=str(tmp_path / "logs"))
    graph = build_unit_graph(ws, tracer, run_id)
    state = graph.invoke(
        {
            "task": "sum the v column",
            "unit_index": 0,
            "goal": "compute total of v",
            "exploration_budget": 1,
            "evidence_summary": "",
            "prior_observations": "",
            "decisions": [],
            "code": "",
            "exec_stdout": "",
            "exec_stderr": "",
            "attempt": 0,
            "feedback": "",
            "report": {},
            "status": "running",
        }
    )
    tracer.close()
    return ws, state


def test_commit_path(fake_env, tmp_path):
    ws, state = _run_unit(fake_env, tmp_path, "v" + chr(10) + "1" + chr(10) + "2" + chr(10) + "3")
    assert state["status"] == "committed"
    assert "6" in state["exec_stdout"]
    assert len(ws.repo.heads) >= 1
    assert (ws.root / "unit0.py").exists()


def test_failure_path_leaves_zero_trace(fake_env, tmp_path, monkeypatch):
    # make generated code always crash
    def crashing_ask(prompt):
        if "Condense" in prompt:
            return "obs"
        return "```python" + chr(10) + "raise RuntimeError('boom')" + chr(10) + "```"

    monkeypatch.setattr(fake_env, "ask", crashing_ask)
    ws, state = _run_unit(fake_env, tmp_path, "v" + chr(10) + "1")
    assert state["status"] == "failed"
    assert not (ws.root / "unit0.py").exists()  # rolled back: zero trace