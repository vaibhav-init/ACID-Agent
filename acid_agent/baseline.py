"""Baseline ReAct agent — the ablation control WITHOUT any ACID machinery.

Same backbone LLM, same workspace, one python-execution tool, plain ReAct loop.
Whatever code it writes stays written (no staging, no validation gate, no rollback):
exactly the failure mode the paper's transactional design fixes.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from .llm import get_backbone

BASELINE_SYSTEM = """You are a data-analysis agent working on CSV files inside the
current directory. Use the run_python tool to explore data and compute answers.
Print intermediate results so you can inspect them. When you know the final answer,
reply with just the answer (include key numbers)."""


def build_baseline_agent(ws, tracer, run_id):
    @tool
    def run_python(code: str) -> str:
        """Execute python code in the task workspace. print() anything you want to see."""
        result = ws.run_code(code, name="baseline_snippet.py", timeout_s=180)
        tracer.log("baseline_exec", run_id=str(run_id), ok=result.ok)
        out = (result.stdout + chr(10) + result.stderr).strip()
        return out[-4000:] if out else "(no output)"

    return create_react_agent(get_backbone(), [run_python], prompt=BASELINE_SYSTEM)


def run_baseline(task: str, ws, tracer, run_id, max_steps: int = 25) -> str:
    agent = build_baseline_agent(ws, tracer, run_id)
    state = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": max_steps * 4},
    )
    answer = state["messages"][-1].content
    tracer.log("final_answer", agent="baseline", answer=str(answer)[:2000])
    return str(answer)