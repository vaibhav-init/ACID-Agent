"""Single entry point to run one task through either agent."""

import uuid

from .baseline import run_baseline
from .config import get_conn
from .graphs.task_graph import build_task_graph
from .tracer import Tracer
from .workspace import Workspace


def run_task(task_text: str, agent_type: str = "acid", seed_files: dict | None = None, slug: str | None = None) -> str:
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

        if agent_type == "acid":
            graph = build_task_graph(ws, tracer, run_id)
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
        else:
            answer = run_baseline(task_text, ws, tracer, run_id)

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