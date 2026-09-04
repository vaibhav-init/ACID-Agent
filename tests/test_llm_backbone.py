"""Backbone gateway dispatch tests (llm.py / claude_runner.py).

The gateway must route identically-shaped calls through either CLI:
  claude   -> `claude -p ...` (default)
  opencode -> `opencode run -m <model>` (Qwen family, regime experiments)
"""

import subprocess
from types import SimpleNamespace

import pytest

from acid_agent import claude_runner as cr
from acid_agent import llm


def _settings(**over):
    base = dict(
        backbone="opencode",
        opencode_bin="opencode",
        opencode_model="opencode-go/qwen3.8-flash",
        claude_bin="claude",
        claude_model="",
        claude_timeout_s=900,
    )
    return SimpleNamespace(**{**base, **over})


def test_strip_ansi():
    dirty = "\x1b[0m\x1b[1m> build · m\x1b[0m\nOK\x1b[0m"
    assert llm.strip_ansi(dirty) == "> build · m\nOK"


def test_opencode_ask_argv(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example.com")
    captured = {}

    def fake_run(args, **kw):
        captured["args"], captured["kw"] = args, kw
        return subprocess.CompletedProcess(args, 0, stdout="OK", stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    assert llm.ask("hi") == "OK"
    assert captured["args"] == ["opencode", "run", "-m", "opencode-go/qwen3.8-flash", "hi"]
    assert captured["kw"]["cwd"] is None
    assert "ANTHROPIC_BASE_URL" not in captured["kw"]["env"]  # no base-URL leakage
    assert captured["kw"]["stdin"] == subprocess.DEVNULL


def test_opencode_isolated_uses_plan_agent_and_sandbox(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    captured = {}

    def fake_run(args, **kw):
        captured["args"], captured["kw"] = args, kw
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    llm.ask_isolated("hi")
    assert "--agent" in captured["args"] and "plan" in captured["args"]
    assert "acid_isolated_" in captured["kw"]["cwd"]


def test_opencode_rc_failure_raises(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        llm.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 3, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="opencode CLI failed"):
        llm.ask("hi")


def test_claude_path_unchanged_when_backbone_claude(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(backbone="claude"))
    captured = {}

    def fake_run(args, **kw):
        captured["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="OK", stderr="")

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    assert llm.ask("hi") == "OK"
    assert captured["args"][:3] == ["claude", "-p", "hi"]


def test_run_claude_routes_to_opencode_session(monkeypatch, tmp_path):
    monkeypatch.setattr(cr, "get_settings", lambda: _settings())
    captured = {}

    def fake_run(args, **kw):
        captured["args"], captured["kw"] = args, kw
        return subprocess.CompletedProcess(args, 0, stdout="\x1b[0mdone\x1b[0m", stderr="")

    monkeypatch.setattr(cr.subprocess, "run", fake_run)
    res = cr.run_claude("write unit0.py", cwd=tmp_path)
    assert res.ok and res.stdout == "done"
    assert captured["args"][0] == "opencode" and "-m" in captured["args"]
    assert "--dir" in captured["args"] and str(tmp_path) in captured["args"]


def test_run_claude_opencode_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(cr, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        cr.subprocess, "run",
        lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("opencode", 1)),
    )
    res = cr.run_claude("x", cwd=tmp_path, timeout_s=5)
    assert not res.ok and res.stderr == "OPENCODE TIMEOUT"
