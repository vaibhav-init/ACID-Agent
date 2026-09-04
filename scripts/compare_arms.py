"""Render the A/B comparison between the two agent arms from results/*.json.

Reads the newest result file per agent (synthetic `eval_*` or KramaBench `krama_*`,
whichever suite is asked for) and prints score + the paper's consistency metric
side by side. Tolerates checkpointed/partial files so it can be run mid-eval.

Usage:
  python scripts/compare_arms.py                 # synthetic suite
  python scripts/compare_arms.py --suite krama   # real KramaBench
"""

import json
import sys
from pathlib import Path
from statistics import mean, variance

RESULTS = Path(__file__).resolve().parents[1] / "results"
ARMS = ["claude", "claude-acid"]  # baseline first, then the transactional arm


def _newest(prefix: str, agent: str) -> Path | None:
    # agent names overlap ("claude" is a prefix of "claude-acid"), so match the
    # exact agent field inside the file rather than trusting the filename.
    cands = []
    for p in RESULTS.glob(f"{prefix}_*.json"):
        try:
            if json.loads(p.read_text()).get("agent") == agent:
                cands.append(p)
        except Exception:
            continue
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _consistency(raw: dict) -> float:
    """Paper metric: sqrt(mean per-task variance across runs). Needs >=2 runs."""
    vs = [variance(v) for v in raw.values() if len(v) > 1]
    return (sum(vs) / len(vs)) ** 0.5 if vs else float("nan")


def main() -> int:
    suite = "krama" if "--suite" in sys.argv and "krama" in sys.argv else "eval"
    label = "KramaBench" if suite == "krama" else "synthetic"

    loaded = {}
    for agent in ARMS:
        p = _newest(suite, agent)
        if p:
            loaded[agent] = (json.loads(p.read_text()), p)

    if not loaded:
        print(f"no {label} results in {RESULTS}/ yet")
        return 1

    tasks = sorted({t for d, _ in loaded.values() for t in d.get("raw_scores", {})})
    w = max([len(t) for t in tasks] + [12])

    print(f"\n{label} suite — A/B\n" + "=" * (w + 42))
    print(f"{'task':<{w}}  {'baseline':>14}  {'acid':>14}  {'delta':>7}")
    print("-" * (w + 42))
    for t in tasks:
        cells, vals = [], {}
        for agent in ARMS:
            raw = loaded.get(agent, ({}, None))[0].get("raw_scores", {}).get(t) or []
            vals[agent] = mean(raw) if raw else None
            # fixed-width cells so the table stays aligned once both arms fill in
            cells.append(f"{mean(raw)*100:>4.0f}% (n={len(raw)})".rjust(14) if raw else "—".rjust(14))
        b, a = vals["claude"], vals["claude-acid"]
        delta = f"{(a - b) * 100:>+6.0f}%" if b is not None and a is not None else f"{'—':>7}"
        print(f"{t:<{w}}  {cells[0]}  {cells[1]}  {delta}")
    print("-" * (w + 42))

    print()
    for agent in ARMS:
        if agent not in loaded:
            print(f"{agent:<12} (no results yet)")
            continue
        d, p = loaded[agent]
        raw = d.get("raw_scores", {})
        n = sum(len(v) for v in raw.values())
        c = _consistency(raw)
        cs = "n/a (needs >=2 runs)" if c != c else f"{c:.3f}"
        print(f"{agent:<12} score={d.get('overall_score'):>5}%  consistency={cs:<20} runs={n}  <- {p.name}")

    if all(a in loaded for a in ARMS):
        cb, ca = (_consistency(loaded[a][0].get("raw_scores", {})) for a in ARMS)
        if cb == cb and ca == ca:
            # lower consistency value = less run-to-run variance = better
            verdict = "ACID more consistent" if ca < cb else ("baseline more consistent" if ca > cb else "tied")
            print(f"\nconsistency (lower=better): baseline {cb:.3f} vs acid {ca:.3f}  ->  {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
