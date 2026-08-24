"""Task-level supervisor graph.

START -> plan_units -> [run_unit -> more?] -> ... -> assemble_answer -> END

The supervisor owns the budgets from the paper:
  * <= MAX_UNITS semantic transaction units per task
  * exploration budget decays by one round every two units
Each unit runs the atomic unit_graph; only COMMITTED unit outputs flow forward.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .. import memory as memory_mod
from ..config import get_conn, get_settings
from ..llm import ask, ask_structured
from ..workspace import Workspace
from .unit_graph import build_unit_graph


class UnitPlan(BaseModel):
    goals: list[str] = []


class Completion(BaseModel):
    complete: bool = False
    reason: str = ""


class TaskState(TypedDict):
    task: str
    goals: list[str]
    unit_index: int            # next unit to run
    evidence_summary: str      # validated evidence carried across units
    results: list[dict]        # committed unit summaries
    final_answer: str


PLAN_PROMPT = """Break this data-analysis task into {max_units} or fewer sequential
atomic units. Each unit must be independently executable with pandas on local CSVs,
have a clear verifiable goal, and build toward the final answer. Prefer 2-5 units.

Task: {task}

Known memory from past tasks:
{memory}"""


def build_task_graph(ws: Workspace, tracer, run_id):
    s = get_settings()
    unit_graph = build_unit_graph(ws, tracer, run_id)

    def plan_units(state: TaskState) -> dict:
        plan: UnitPlan = ask_structured(
            PLAN_PROMPT.format(
                max_units=min(5, s.max_units),
                task=state["task"],
                memory=memory_mod.context_block(),
            ),
            UnitPlan,
        )
        goals = plan.goals[: s.max_units] or ["Answer the task directly from the data."]
        tracer.log("plan", goals=goals)
        return {"goals": goals, "unit_index": 0}

    def run_unit(state: TaskState) -> dict:
        i = state["unit_index"]
        goal = state["goals"][i]
        budget = max(s.exploration_min_rounds, s.exploration_max_rounds - i // 2)
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO units (run_id, unit_index, goal) VALUES (%s,%s,%s)",
                (run_id, i, goal),
            )
        tracer.log("unit_start", unit_index=i, goal=goal, exploration_budget=budget)

        result = unit_graph.invoke(
            {
                "task": state["task"],
                "unit_index": i,
                "goal": goal,
                "exploration_budget": budget,
                "evidence_summary": state["evidence_summary"],
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
        results = list(state["results"])
        evidence = state["evidence_summary"]
        if result["status"] == "committed":
            results.append({"unit": i, "goal": goal, "output": result["exec_stdout"][-800:]})
            # validated observations flow forward; failed units contribute nothing
            evidence = (evidence + chr(10) + f"[unit {i}: {goal}] " + result["exec_stdout"][-500:]).strip()
        else:
            results.append({"unit": i, "goal": goal, "output": "(unit failed; excluded)"})
        return {"results": results, "evidence_summary": evidence, "unit_index": i + 1}

    def more_units(state: TaskState) -> str:
        if state["unit_index"] >= len(state["goals"]):
            return "assemble"
        if state["unit_index"] >= s.max_units:
            return "assemble"
        # early completion check every unit (cheap)
        try:
            comp: Completion = ask_structured(
                f"""Task: {state['task']}

Committed unit results so far:
{chr(10).join(str(r) for r in state['results'])}

Is the task fully answered by these results?""",
                Completion,
            )
            if comp.complete:
                return "assemble"
        except Exception:
            pass
        return "run_unit"

    def assemble_answer(state: TaskState) -> dict:
        answer = ask(f"""You are finalizing a data-analysis task.

Original task: {state['task']}

Committed (validated) unit results:
{chr(10).join(f"- unit {r['unit']} ({r['goal']}): {r['output']}" for r in state['results'])}

Produce the final answer to the original task. Be precise and concise;
include the key number(s). If some units failed, answer with what succeeded.""")
        tracer.log("final_answer", answer=answer[:2000])
        return {"final_answer": answer}

    g = StateGraph(TaskState)
    g.add_node("plan_units", plan_units)
    g.add_node("run_unit", run_unit)
    g.add_node("assemble_answer", assemble_answer)

    g.add_edge(START, "plan_units")
    g.add_edge("plan_units", "run_unit")
    g.add_conditional_edges("run_unit", more_units, {"run_unit": "run_unit", "assemble": "assemble_answer"})
    g.add_edge("assemble_answer", END)
    return g.compile()