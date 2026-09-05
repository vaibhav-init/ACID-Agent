"""CLI entry points.

Usage:
  python scripts/acid_cli.py run-task "What is total revenue for north?" --agent acid
  python scripts/acid_cli.py run-task "..." --agent baseline --data-dir path/to/csvs
  python scripts/acid_cli.py run-task "..." --agent react --data-dir path/to/csvs
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
from acid_agent.eval.kramabench_tasks import load_domain, grade_task, get_available_domains, is_gradeable, DOMAINS
from acid_agent.runner import run_task

app = typer.Typer(help="ACID-Agent CLI")


@app.command("run-task")
def run_task_cmd(
    task: str = typer.Argument(..., help="Task text"),
    agent: str = typer.Option("acid", help="baseline | react | acid"),
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
    agent: str = typer.Option("acid", help="baseline | react | acid"),
    runs: int = typer.Option(3, help="Runs per task"),
    synthetic: bool = typer.Option(False, help="Use synthetic tasks instead of KramaBench"),
    domain: str = typer.Option("archeology", help="KramaBench domain"),
    task_ids: str = typer.Option("all", help="Task indices: all, 0-5, 0,1,2"),
    tag: str = typer.Option("", help="Label for this configuration; isolates workspaces/memory so a "
                                     "re-run under different settings cannot inherit the last one"),
):
    if synthetic:
        # Use built-in synthetic tasks
        from acid_agent.eval.kramabench import builtin_tasks
        def runner(t, run):
            # _r{run} keeps repetitions independent (own workspace + own memory scope)
            return run_task(t.question, agent_type=agent, seed_files=t.seed_files, slug=f"eval_{agent}_{t.id}_r{run}")
        evaluate(agent, runner, n_runs=runs)
    else:
        # Use real KramaBench tasks
        _run_kramabench_eval(agent, domain, runs, task_ids, tag)


def _run_kramabench_eval(agent: str, domain: str, runs: int, task_ids: str, tag: str = ""):
    """Run evaluation on real KramaBench tasks."""
    tasks = load_domain(domain)
    
    # Filter by task_ids
    if task_ids != "all":
        if "-" in task_ids:
            start, end = map(int, task_ids.split("-"))
            tasks = tasks[start:end+1]
        else:
            indices = list(map(int, task_ids.split(",")))
            tasks = [tasks[i] for i in indices if 0 <= i < len(tasks)]
    
    skipped = [t.id for t in tasks if not is_gradeable(t)]
    tasks = [t for t in tasks if is_gradeable(t)]
    if skipped:
        typer.echo(f"Skipping {len(skipped)} task(s) with un-judgeable answer types "
                   f"(would score a fake 0.00): {', '.join(skipped)}")

    typer.echo(f"Running {len(tasks)} KramaBench tasks ({domain}) with agent={agent}, runs={runs}")
    
    import json, time
    from pathlib import Path
    from statistics import mean, variance
    scores: dict[str, list[float]] = {t.id: [] for t in tasks}

    # A tag isolates a re-run: Workspace.create reuses an existing directory and
    # commits whatever it finds, so an untagged repeat would inherit the previous
    # run's unit*.py instead of starting clean.
    suffix = f"-{tag}" if tag else ""
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"krama_{agent}{suffix}_{domain}_{int(time.time())}.json"

    def _summary() -> dict:
        """Safe on partially filled scores, so it can checkpoint mid-eval."""
        done = {k: v for k, v in scores.items() if v}
        per_task = {k: mean(v) for k, v in done.items()}
        alls = [s for v in done.values() for s in v]
        overall = mean(alls) if alls else 0.0
        variances = [variance(v) for v in done.values() if len(v) > 1]
        consistency = (sum(variances) / len(variances)) ** 0.5 if variances else 0.0
        return {
            "agent": agent, "domain": domain, "runs": runs, "tag": tag,
            "runs_completed": len(alls),
            "overall_score": round(overall * 100, 1),
            "consistency_sqrt_avg_var": round(consistency, 3),
            "per_task": {k: round(v * 100, 1) for k, v in per_task.items()},
            "raw_scores": scores,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    for t in tasks:
        for run in range(runs):
            try:
                answer = run_task(
                    t.question,
                    agent_type=agent,
                    seed_files=t.seed_files,
                    slug=f"krama_{agent}{suffix}_{t.id}_r{run}",
                )
            except Exception as e:
                typer.echo(f"[{agent}] {t.id} run {run}: ERROR {e}")
                answer = ""
            sc = grade_task(t, str(answer))
            if sc is None:  # not judgeable; never counted as a failure
                continue
            scores[t.id].append(sc)
            typer.echo(f"[{agent}] {t.id} run {run}: score={sc:.2f}", nl=True)
            # Checkpoint every run — the ACID arm takes ~15 min/run, so an
            # interrupted eval must not throw away completed work.
            path.write_text(json.dumps(_summary(), indent=2))

    summary = _summary()
    per_task = {k: v / 100 for k, v in summary["per_task"].items()}
    overall = summary["overall_score"] / 100
    consistency = summary["consistency_sqrt_avg_var"]

    typer.echo(f"\n{'='*60}")
    typer.echo(f"KramaBench Results: {agent} on {domain}")
    typer.echo(f"{'='*60}")
    typer.echo(f"Overall Score: {overall*100:.1f}%")
    typer.echo(f"Consistency (sqrt avg per-task variance): {consistency:.3f}")
    typer.echo(f"Per-task scores:")
    for tid, sc in per_task.items():
        typer.echo(f"  {tid}: {sc*100:.1f}%")

    path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\nResults saved to {path}")


@app.command("kramabench")
def kramabench_cmd(
    action: str = typer.Argument(..., help="run | compare | domains"),
    agent: str = typer.Option("acid", help="baseline | react | acid"),
    domain: str = typer.Option("archeology", help="KramaBench domain"),
    runs: int = typer.Option(3, help="Runs per task"),
    task_ids: str = typer.Option("all", help="Task indices"),
    tag: str = typer.Option("", help="Label for this configuration (isolates workspaces/memory)"),
):
    """KramaBench evaluation commands."""
    if action == "domains":
        typer.echo("Available KramaBench domains:")
        for d in DOMAINS:
            available = "✓" if d in get_available_domains() else "✗ (data not downloaded)"
            typer.echo(f"  {d}: {available}")
    elif action == "run":
        _run_kramabench_eval(agent, domain, runs, task_ids, tag)
    elif action == "compare":
        typer.echo("Run both agents and compare:")
        typer.echo(f"  python scripts/acid_cli.py eval --agent acid --domain {domain} --runs {runs}")
        typer.echo(f"  python scripts/acid_cli.py eval --agent baseline --domain {domain} --runs {runs}")
        typer.echo(f"  python scripts/acid_cli.py eval --agent react --domain {domain} --runs {runs}")


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