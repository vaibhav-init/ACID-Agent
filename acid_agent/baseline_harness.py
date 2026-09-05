"""Raw agent baseline — the ablation control without ACID machinery.

Same model, same workspace, ONE headless session.
NO transactional rollback, NO confidence validation, NO exploration cycles:
exactly the failure mode the paper's transactional design fixes.
"""

from .opencode_runner import run_opencode_session

BASELINE_PROMPT = """You are a data-analysis agent working on files in the current directory.
Analyze the data files and answer the user's question.
Print your final answer clearly at the end.
"""


def run_baseline_harness(task: str, ws, tracer=None, run_id=None) -> str:
    prompt = f"{BASELINE_PROMPT}\n\nTask: {task}"
    result = run_opencode_session(prompt, cwd=ws.root)
    if tracer:
        tracer.log(
            "baseline_exec",
            run_id=str(run_id) if run_id else None,
            ok=result.ok,
            stdout=result.stdout[-1500:],
            stderr=result.stderr[-800:],
        )
    answer = result.stdout.strip() if result.ok else ""
    return answer or result.stderr.strip()
