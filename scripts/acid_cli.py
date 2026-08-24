"""CLI entry points.

Usage:
  python scripts/acid_cli.py run-task "What is total revenue for north?" --agent acid
  python scripts/acid_cli.py run-task "..." --agent baseline --data-dir path/to/csvs
  python scripts/acid_cli.py eval --agent acid --runs 3
  python scripts/acid_cli.py skill-register avg_monthly_revenue "Computes average monthly revenue" 
  python scripts/acid_cli.py skill-validate avg_monthly_revenue
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from acid_agent import skills
from acid_agent.eval.kramabench import evaluate
from acid_agent.runner import run_task

app = typer.Typer(help="ACID-Agent CLI")


@app.command("run-task")
def run_task_cmd(
    task: str = typer.Argument(..., help="Task text"),
    agent: str = typer.Option("acid", help="acid | baseline"),
    data_dir: str = typer.Option(None, help="Optional dir of CSVs to seed the workspace"),
    slug: str = typer.Option(None, help="Workspace name"),
):
    seed = {}
    if data_dir:
        d = Path(data_dir)
        for f in sorted(d.iterdir()):
            if f.is_file():
                seed[f.name] = f.read_text(encoding="utf-8", errors="ignore")
    answer = run_task(task, agent_type=agent, seed_files=seed or None, slug=slug)
    typer.echo("=== FINAL ANSWER ===")
    typer.echo(answer)


@app.command("eval")
def eval_cmd(
    agent: str = typer.Option("acid", help="acid | baseline"),
    runs: int = typer.Option(3, help="Runs per task"),
):
    from acid_agent.eval.kramabench import builtin_tasks

    def runner(t):
        return run_task(t.question, agent_type=agent, seed_files=t.seed_files, slug=f"eval_{agent}_{t.id}")

    evaluate(agent, runner, n_runs=runs)


@app.command(name="skill-register")
def skill_register(name: str, description: str):
    skills.register_skill(name, description, str(skills.SKILLS_DIR / name))
    typer.echo(f"registered {name}")


@app.command(name="skill-validate")
def skill_validate(name: str):
    ok = skills.validate_skill(name)
    typer.echo(f"{name}: {'PASS' if ok else 'FAIL'}")
    raise typer.Exit(0 if ok else 1)


@app.command(name="skill-invoke")
def skill_invoke(name: str, args: list[str] = typer.Argument(default=[])):
    typer.echo(skills.invoke_skill(name, list(args)))


if __name__ == "__main__":
    app()