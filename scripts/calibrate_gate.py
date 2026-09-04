"""Calibrate the code-span gate threshold from observed runs.

The paper states `span_divergence_min = 0.50`, but that number is not portable
across scorers: measured on Qwen3-0.6B-Base, a script that correctly implements
every evidence item scored a max divergence of 0.30-0.43, so the paper rule
rejects well-grounded code and exhausts the retry budget on every unit.

This reads the signals that were actually recorded and reports what each
candidate threshold would do. Feed it runs made with GATE_BYPASS=1: the bypass
forces PASS so a run completes normally, while `review_decision` still records
the verdict the gate WOULD have reached and both metrics are persisted.

Usage:
  python scripts/calibrate_gate.py --tag calib
  python scripts/calibrate_gate.py --since-min 300 --agent claude-acid
"""

import argparse
from statistics import mean, median

from acid_agent.config import get_conn, get_settings


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="claude-acid")
    ap.add_argument("--tag", default="", help="match runs whose slug carries this --tag")
    ap.add_argument("--since-min", type=int, default=1440)
    a = ap.parse_args()
    cfg = get_settings()

    where = ["r.agent_type = %s", "r.started_at > now() - make_interval(mins => %s)"]
    params: list = [a.agent, a.since_min]
    if a.tag:
        # slugs are krama_<agent>-<tag>_<task>_r<n>
        where.append("r.task_slug LIKE %s")
        params.append(f"%-{a.tag}\\_%")

    with get_conn() as c:
        has_slug = c.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='runs' AND column_name='task_slug'"
        ).fetchone()
        if not has_slug and a.tag:
            print("runs table has no task_slug column; ignoring --tag")
            where.pop()
            params.pop()

        rows = c.execute(
            f"""SELECT v.max_span_divergence, v.max_span_surprise, v.contrast_min_ratio,
                       v.decision_surprise, v.review_decision, v.execution_ok,
                       v.reflection_ok, v.gate_semantics, r.run_id
                FROM validations v JOIN runs r ON r.run_id = v.run_id
                WHERE {' AND '.join(where)}
                ORDER BY v.id""",
            tuple(params),
        ).fetchall()

    if not rows:
        print(f"no {a.agent} gate attempts found in the last {a.since_min} min")
        return 1

    div = [r[0] for r in rows if r[0] is not None]
    sur = [r[1] for r in rows if r[1] is not None]
    ratio = [r[2] for r in rows if r[2] is not None]
    n_runs = len({r[8] for r in rows})

    print(f"{len(rows)} gate attempts across {n_runs} run(s)\n")
    if not div:
        print("No span metrics recorded -> the local scorer never loaded. "
              "Check `pip install -e '.[confidence]'` and CONFIDENCE_DEVICE.")
        return 1

    print("span DIVERGENCE (paper metric; retry BELOW threshold)")
    print(f"  n={len(div)}  min={min(div):.3f}  p10={_pct(div,.10):.3f}  "
          f"p25={_pct(div,.25):.3f}  median={median(div):.3f}  "
          f"p75={_pct(div,.75):.3f}  max={max(div):.3f}  mean={mean(div):.3f}")
    print("\nspan SURPRISE (reference metric; retry ABOVE threshold)")
    print(f"  n={len(sur)}  min={min(sur):.3f}  median={median(sur):.3f}  "
          f"max={max(sur):.3f}  mean={mean(sur):.3f}")
    if ratio:
        print("\ncontrast ratio (both agree; retry BELOW 0.25)")
        print(f"  n={len(ratio)}  min={min(ratio):.3f}  median={median(ratio):.3f}")
    else:
        print("\ncontrast ratio: no probes -> no evidence/code conflict was flagged")

    print("\n" + "-" * 62)
    print("what each PAPER threshold would reject")
    print("-" * 62)
    print(f"{'threshold':>10} {'rejected':>10} {'rate':>8}")
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        n = sum(1 for d in div if d < t)
        flag = "   <- paper's stated value" if abs(t - 0.50) < 1e-9 else ""
        print(f"{t:>10.2f} {n:>7}/{len(div)} {n/len(div)*100:>7.0f}%{flag}")

    print(f"\n{'threshold':>10} {'rejected':>10} {'rate':>8}   (reference rule)")
    for t in (0.10, 0.25, 0.50, 1.00):
        n = sum(1 for s in sur if s > t)
        print(f"{t:>10.2f} {n:>7}/{len(sur)} {n/len(sur)*100:>7.0f}%")

    # A gate that never fires and a gate that always fires are equally useless.
    # Target a threshold that rejects a minority of attempts.
    target = next((t for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
                   if 0.10 <= sum(1 for d in div if d < t) / len(div) <= 0.35), None)
    print("\n" + "=" * 62)
    if target is None:
        lo = sum(1 for d in div if d < 0.05) / len(div)
        print("No candidate threshold rejects 10-35% of attempts.")
        print(f"  At the loosest (0.05) it already rejects {lo*100:.0f}%.")
        print("  The metric does not separate on this data: either widen the run set")
        print("  or run gate_semantics=reference, whose signal is on a different scale.")
    else:
        rate = sum(1 for d in div if d < target) / len(div)
        print(f"SUGGESTED  span_divergence_min = {target:.2f}   "
              f"(rejects {rate*100:.0f}% of attempts)")
        print(f"  paper's 0.50 would reject "
              f"{sum(1 for d in div if d < 0.50)/len(div)*100:.0f}% -- "
              "with max_retries_per_unit=2 that fails most units outright.")
    print("=" * 62)
    print(f"\ncurrent settings: gate_semantics={cfg.gate_semantics}  "
          f"span_divergence_min={cfg.span_divergence_min}  "
          f"span_surprise_retry={cfg.span_surprise_retry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
