"""Single entry point to run one task through any agent type.

Agent types:
  claude       — raw Claude Code baseline (the harness, no ACID machinery)
  claude-react — DA-Agent style ReAct loop on the same model, no harness
  claude-acid — Claude Code execution inside the ACID transaction loop
  acid        — alias of claude-acid
"""

import uuid

from .baseline_claude import run_baseline_claude
from .baseline_react import run_baseline_react
from .config import get_conn
from .graphs.task_graph import build_task_graph
from .tracer import Tracer
from .tracing import traced
from .workspace import Workspace


@traced("acid_task", drop=("seed_files",))
def run_task(task_text: str, agent_type: str = "claude-acid", seed_files: dict | None = None, slug: str | None = None) -> str:
    """Returns the final answer string. Creates workspace + DB records + trace."""
    run_id = uuid.uuid4()
    tracer = Tracer(run_id)
    try:
        ws = Workspace.create(
            "workspaces", slug or f"task_{run_id.hex[:8]}", seed_files=seed_files
        )
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, task_text, agent_type) VALUES (%s,%s,%s)",
                (run_id, task_text, agent_type),
            )

        if agent_type in ("claude-acid", "acid"):
            # claude is the execution backend inside explore/generate_execute;
            # the transaction gate (validate + git commit-or-rollback) wraps it.
            graph = build_task_graph(ws, tracer, run_id, task_slug=slug)
            final_state = graph.invoke(
                {
                    "task": task_text,
                    "goals": [],
                    "unit_index": 0,
                    "evidence_summary": "",
                    "results": [],
                    "final_answer": "",
                }
            )
            answer = final_state["final_answer"]
        elif agent_type == "claude":
            answer = run_baseline_claude(task_text, ws, tracer, run_id)
        elif agent_type in ("claude-react", "react"):
            # Same model, no Claude Code harness: this module owns the loop and
            # the action execution, so the harness contributes nothing.
            answer = run_baseline_react(task_text, ws, tracer, run_id)
        else:
            raise ValueError(
                f"unknown agent_type: {agent_type!r} "
                "(use claude | claude-react | claude-acid)"
            )

        with get_conn() as conn:
            conn.execute(
                "UPDATE runs SET status='done', final_answer=%s, finished_at=now() WHERE run_id=%s",
                (answer[:4000], run_id),
            )
        return answer
    except Exception:
        try:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE runs SET status='failed', finished_at=now() WHERE run_id=%s",
                    (run_id,),
                )
        except Exception:
            pass
        raise
    finally:
        tracer.close()
