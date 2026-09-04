"""Re-score completed runs from Postgres using the CURRENT graders.

The expensive part of an eval is the agent runs; grading is a pure function of
the stored `final_answer`. When a grader bug is found (see `_num_grader` and
`_norm`), this re-scores history instead of costing another multi-hour eval.

Usage:
  python scripts/regrade.py --suite synthetic
  python scripts/regrade.py --suite krama --domain archeology
  python scripts/regrade.py --suite krama --domain archeology --write
"""

import argparse
import json
import time
from pathlib import Path
from statistics import mean, variance

from acid_agent.config import get_conn

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _tasks(suite: str, domain: str):
    """{question: (task_id, grader_fn)} for the chosen suite."""
    if suite == "synthetic":
        from acid_agent.eval.kramabench import builtin_tasks
        return {t.question: (t.id, t.grader) for t in builtin_tasks()}
    from acid_agent.eval.kramabench_tasks import grade_task, load_domain
    return {
        t.question: (t.id, (lambda a, _t=t: grade_task(_t, a)))
        for t in load_domain(domain)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["synthetic", "krama"], default="synthetic")
    ap.add_argument("--domain", default="archeology")
    ap.add_argument("--agents", default="claude,claude-acid")
    ap.add_argument("--since-min", type=int, default=1440, help="look back this many minutes")
    ap.add_argument("--write", action="store_true", help="save regraded_*.json into results/")
    a = ap.parse_args()

    tasks = _tasks(a.suite, a.domain)
    out = {}
    for agent in a.agents.split(","):
        with get_conn() as c:
            rows = c.execute(
                """SELECT task_text, final_answer FROM runs
                   WHERE agent_type=%s AND status='done'
                     AND started_at > now() - make_interval(mins => %s)
                   ORDER BY started_at""",
                (agent, a.since_min),
            ).fetchall()
        scores: dict[str, list[float]] = {}
        for q, ans in rows:
            hit = tasks.get(q)
            if hit:
                tid, grader = hit
                sc = grader(str(ans or ""))
                if sc is None:
                    continue  # un-judgeable answer type; excluded, not failed
                scores.setdefault(tid, []).append(sc)
        if scores:
            out[agent] = scores

    if not out:
        print("no matching completed runs found")
        return 1

    for agent, scores in out.items():
        alls = [s for v in scores.values() for s in v]
        vs = [variance(v) for v in scores.values() if len(v) > 1]
        cons = (sum(vs) / len(vs)) ** 0.5 if vs else float("nan")
        cs = "n/a" if cons != cons else f"{cons:.3f}"
        print(f"\n{agent}  ({len(alls)} runs)")
        for tid, v in sorted(scores.items()):
            print(f"  {tid:24} {[round(x, 2) for x in v]}  mean={mean(v)*100:5.1f}%")
        print(f"  {'OVERALL':24} {mean(alls)*100:5.1f}%   consistency={cs} (lower=better)")

        if a.write:
            RESULTS.mkdir(exist_ok=True)
            p = RESULTS / f"regraded_{agent}_{a.suite}_{int(time.time())}.json"
            p.write_text(json.dumps({
                "agent": agent, "suite": a.suite, "domain": a.domain,
                "regraded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overall_score": round(mean(alls) * 100, 1),
                "consistency_sqrt_avg_var": None if cons != cons else round(cons, 3),
                "per_task": {k: round(mean(v) * 100, 1) for k, v in scores.items()},
                "raw_scores": scores,
            }, indent=2))
            print(f"  saved -> {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
