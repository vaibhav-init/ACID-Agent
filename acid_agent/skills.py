"""Skill hub (offline atomicity/consistency).

Skills are folders under skills/<name>/ with:
  SKILL.md   — description + usage
  skill.py   — typer CLI implementing the validated workflow
  test_skill.py — generated/curated tests that must pass before the skill is trusted

Registry lives in Postgres (skills table, pgvector embedding when available).
The router picks a skill for a task via LLM choice over registered skills.
"""

import subprocess
import sys
from pathlib import Path

from .config import get_conn
from .llm import ask_structured
from pydantic import BaseModel

SKILLS_DIR = Path("skills")


class SkillChoice(BaseModel):
    skill: str = ""  # empty => no skill applies
    reason: str = ""


def register_skill(name: str, description: str, path: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO skills (name, description, path) VALUES (%s,%s,%s)
               ON CONFLICT (name) DO UPDATE SET description=EXCLUDED.description,
                   path=EXCLUDED.path""",
            (name, description, path),
        )


def list_skills() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT name, description, path FROM skills ORDER BY name"
        ).fetchall()
    return [{"name": r[0], "description": r[1], "path": r[2]} for r in rows]


ROUTE_PROMPT = """Task:
{task}

Available validated skills:
{skills}

If one skill clearly implements this task's workflow, name it. Otherwise return an
empty skill string."""


def route_skill(task_text: str) -> dict | None:
    """Pick a matching validated skill for the task, or None."""
    skills = list_skills()
    if not skills:
        return None
    listing = chr(10).join(f"- {s['name']}: {s['description']}" for s in skills)
    try:
        choice: SkillChoice = ask_structured(ROUTE_PROMPT.format(task=task_text[:1500], skills=listing), SkillChoice)
    except Exception:
        return None
    if not choice.skill:
        return None
    for s in skills:
        if s["name"] == choice.skill:
            return s
    return None


def validate_skill(name: str) -> bool:
    """Run the skill's pytest suite; a skill is only usable when its tests pass."""
    skill_dir = SKILLS_DIR / name
    test_file = skill_dir / "test_skill.py"
    if not test_file.exists():
        return False
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q"],
        stdin=subprocess.DEVNULL,  # see llm._run_cli
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0


def invoke_skill(name: str, args: list[str]) -> str:
    """Call the skill's typer CLI."""
    entry = SKILLS_DIR / name / "skill.py"
    proc = subprocess.run(
        [sys.executable, str(entry)] + args,
        stdin=subprocess.DEVNULL,  # see llm._run_cli
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + chr(10) + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"skill {name} failed: {out[-2000:]}")
    with get_conn() as conn:
        conn.execute("UPDATE skills SET use_count = use_count + 1 WHERE name = %s", (name,))
    return out.strip()