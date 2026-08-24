"""Transactional workspace lifecycle: create -> dirty -> rollback / commit."""

from acid_agent.workspace import Workspace


def test_create_seeds_and_commits(tmp_path):
    ws = Workspace.create(tmp_path, "t1", seed_files={"a.csv": "x,y" + chr(10) + "1,2"})
    assert (ws.root / "a.csv").exists()
    assert len(ws.repo.heads) == 1


def test_rollback_erases_failed_attempt(tmp_path):
    ws = Workspace.create(tmp_path, "t2", seed_files={"data.csv": "v" + chr(10) + "1"})
    before = ws.head()
    # simulate a failed attempt writing junk
    (ws.root / "junk.py").write_text("print('bad attempt')")
    (ws.root / "data.csv").write_text("v" + chr(10) + "CORRUPTED")
    ws.rollback()
    assert ws.head() == before
    assert not (ws.root / "junk.py").exists()
    assert "1" in (ws.root / "data.csv").read_text()


def test_commit_persists_validated_work(tmp_path):
    ws = Workspace.create(tmp_path, "t3", seed_files={"data.csv": "v" + chr(10) + "1"})
    before = ws.head()
    (ws.root / "unit0.py").write_text("print('RESULT: 42')")
    sha = ws.commit("unit-0: compute")
    assert sha != before
    assert (ws.root / "unit0.py").exists()


def test_run_code_captures_output_and_errors(tmp_path):
    ws = Workspace.create(tmp_path, "t4")
    ok = ws.run_code("print('hello')")
    assert ok.ok and "hello" in ok.stdout
    bad = ws.run_code("raise ValueError('boom')")
    assert not bad.ok and "boom" in bad.stderr