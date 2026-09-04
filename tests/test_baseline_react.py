"""Action parsing, execution and loop control for the DA-Agent style baseline.

No LLM and no network: `ask` is monkeypatched to replay a scripted sequence of
responses, which is what makes the loop's control flow (repeat guard, parse
retries, step budget, Terminate) testable at all.
"""

import pytest

import acid_agent.baseline_react as react
from acid_agent.baseline_react import Bash, Python, SQL, Terminate, parse_action
from acid_agent.workspace import Workspace


# ------------------------------------------------------------------ parsing

def test_parses_each_action_type():
    assert isinstance(parse_action('Action: Bash(code="ls -l")'), Bash)
    assert isinstance(
        parse_action('Action: Python(file_path="a.py"):\n```python\nprint(1)\n```'), Python
    )
    assert isinstance(
        parse_action('Action: SQL(file_path="d.db", command="SELECT 1", output="direct")'), SQL
    )
    assert isinstance(parse_action('Action: Terminate(output="42")'), Terminate)


def test_parses_nested_parens_and_quotes():
    a = parse_action("""Action: Bash(code="python -c 'print((1+2))'")""")
    assert isinstance(a, Bash) and "(1+2)" in a.code


def test_python_wins_over_bash_when_both_could_match():
    # A fenced python body can contain the substring Bash(code= ; the Python
    # parser must claim it first or the loop executes the wrong thing.
    text = 'Action: Python(file_path="x.py"):\n```python\nprint("Bash(code=\'rm -rf /\')")\n```'
    assert isinstance(parse_action(text), Python)


def test_returns_none_without_an_action():
    assert parse_action("Thought: still thinking about it.") is None
    assert parse_action("") is None


def test_strips_thought_prefix_from_action_segment():
    a = parse_action("Thought: I will list files.\nAction: Bash(code=\"ls\")\nObservation: ...")
    assert isinstance(a, Bash) and a.code == "ls"


# ---------------------------------------------------------------- execution

def test_bash_action_runs_in_workspace(tmp_path):
    ws = Workspace.create(tmp_path, "react", seed_files={"data.csv": "v\n1\n2\n"})
    out, done = react._execute(ws, Bash(code="cat data.csv"), 30)
    assert not done and "1" in out and "2" in out


def test_python_action_writes_and_runs(tmp_path):
    ws = Workspace.create(tmp_path, "react", seed_files={"data.csv": "v\n1\n2\n3\n"})
    out, done = react._execute(
        ws, Python(code="print('TOTAL', 1+2+3)", filepath="./calc.py"), 30
    )
    assert not done and "TOTAL 6" in out
    assert (ws.root / "calc.py").exists()


def test_failing_python_reports_exit_code(tmp_path):
    ws = Workspace.create(tmp_path, "react", seed_files={"d.csv": "x\n"})
    out, _ = react._execute(ws, Python(code="raise RuntimeError('boom')", filepath="b.py"), 30)
    assert "exit code" in out and "boom" in out


def test_terminate_returns_answer_and_done(tmp_path):
    ws = Workspace.create(tmp_path, "react", seed_files={"d.csv": "x\n"})
    out, done = react._execute(ws, Terminate(output="5.28"), 30)
    assert done and out == "5.28"


# --------------------------------------------------------------- loop control

def _scripted(monkeypatch, responses):
    """Replay `responses` in order; record the prompts the loop built."""
    seen = []

    def fake_ask(prompt):
        seen.append(prompt)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    monkeypatch.setattr(react, "ask", fake_ask)
    return seen


def test_loop_terminates_with_answer(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path, "react", seed_files={"data.csv": "v\n1\n2\n3\n"})
    _scripted(monkeypatch, [
        'Thought: sum it.\nAction: Python(file_path="s.py"):\n```python\n'
        'import pandas as pd; print("TOTAL", pd.read_csv("data.csv").v.sum())\n```',
        'Thought: got it.\nAction: Terminate(output="6")',
    ])
    assert react.run_baseline_react("sum v", ws) == "6"


def test_repeated_action_aborts_instead_of_burning_budget(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path, "react", seed_files={"d.csv": "x\n"})
    seen = _scripted(monkeypatch, ['Thought: again.\nAction: Bash(code="ls")'])
    monkeypatch.setattr(react, "ask", lambda p: (seen.append(p), 'Action: Bash(code="ls")')[1])
    react.run_baseline_react("t", ws)
    # nudge once, abort on the second repeat -- far short of react_max_steps
    assert len(seen) < 8, f"loop did not abort on repeats: {len(seen)} calls"


def test_unparseable_responses_give_up_after_retry_limit(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path, "react", seed_files={"d.csv": "x\n"})
    seen = _scripted(monkeypatch, ["Thought: no action here."])
    react.run_baseline_react("t", ws)
    # PARSE_RETRY_LIMIT attempts, then one final-answer call
    assert len(seen) <= react.PARSE_RETRY_LIMIT + 2


def test_budget_exhaustion_still_produces_an_answer(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path, "react", seed_files={"d.csv": "x\n"})
    calls = {"n": 0}

    def fake_ask(prompt):
        calls["n"] += 1
        if "FINAL ANSWER" in prompt:
            return "9.9"
        # never terminates, and varies so the repeat guard doesn't fire
        return f'Thought: step.\nAction: Bash(code="echo {calls["n"]}")'

    monkeypatch.setattr(react, "ask", fake_ask)
    monkeypatch.setattr(react.get_settings(), "react_max_steps", 3)
    assert react.run_baseline_react("t", ws) == "9.9"


def test_prompt_carries_the_action_space_and_task(tmp_path, monkeypatch):
    ws = Workspace.create(tmp_path, "react", seed_files={"d.csv": "x\n"})
    seen = _scripted(monkeypatch, ['Action: Terminate(output="ok")'])
    react.run_baseline_react("compute the median price", ws)
    assert "compute the median price" in seen[0]
    assert "Bash(code=" in seen[0] and "Terminate(output=" in seen[0]
    assert "Observation: You are in the folder now." in seen[0]
