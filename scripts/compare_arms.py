"""Render A/B comparisons from results/*.json.

Groups result files by (agent, tag), MERGES their raw_scores, and prints a
per-task table. Comparisons are only meaningful within one context, so scope
them with --domain (KramaBench) and --tags; the old newest-file-per-agent
behavior juxtaposed runs from different domains and was removed.

Usage:
  python scripts/compare_arms.py                                              # synthetic
  python scripts/compare_arms.py --suite krama --domain archeology
  python scripts/compare_arms.py --suite krama --domain archeology --tags hard7,hard7bypass
"""

import json
from math import comb
from pathlib import Path
from statistics import mean, variance

import typer

RESULTS = Path(__file__).resolve().parents[1] / "results"
ORDER = ["claude", "claude-react", "claude-acid"]  # preferred column order

app = typer.Typer(help="Compare agent arms from results/")


def _load(suite: str, domain: str, tags: list[str]) -> dict[tuple[str, str], dict]:
    """Merge every matching result file into one raw-score dict per (agent, tag)."""
    groups: dict[tuple[str, str], dict] = {}
    for p in RESULTS.glob(f"{suite}_*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if domain and d.get("domain") != domain:
            continue
        if tags and d.get("tag", "") not in tags:
            continue
        agent = d.get("agent")
        if not agent:
            continue
        g = groups.setdefault((agent, d.get("tag", "")), {"raw": {}, "files": []})
        for t, scores in d.get("raw_scores", {}).items():
            g["raw"].setdefault(t, []).extend(scores)
        g["files"].append(p.name)
    return groups


def _consistency(raw: dict) -> float:
    """Paper metric: sqrt(mean per-task variance across runs). Needs >=2 runs."""
    vs = [variance(v) for v in raw.values() if len(v) > 1]
    return (sum(vs) / len(vs)) ** 0.5 if vs else float("nan")


def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p. 2x2 table: arm (a pass, b fail) vs base (c pass, d fail)."""
    n1, n2 = a + b, c + d
    m, n = a + c, a + b + c + d
    if not (n1 and n2 and m and n - m):
        return float("nan")

    def prob(x: int) -> float:
        return comb(m, x) * comb(n - m, n1 - x) / comb(n, n1)

    lo, hi = max(0, n1 - (n - m)), min(n1, m)
    p_obs = prob(a)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))


def _label(k: tuple[str, str]) -> str:
    return f"{k[0]}-{k[1]}" if k[1] else k[0]


@app.command()
def main(
    suite: str = typer.Option("eval", help="Result-file prefix: eval | krama"),
    domain: str = typer.Option("", help="KramaBench domain filter (krama suite)"),
    tags: str = typer.Option("", help="Comma-separated tag filter; empty = all"),
):
    groups = _load(suite, domain, [t for t in tags.split(",") if t])
    if not groups:
        typer.echo(f"no matching {suite} results in {RESULTS}/")
        raise typer.Exit(1)

    keys = sorted(groups, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else len(ORDER), k))
    base = next((k for k in keys if k[0] == "claude"), keys[0])
    base_raw = [s for v in groups[base]["raw"].values() for s in v]

    tasks = sorted({t for g in groups.values() for t in g["raw"]})
    w = max([len(t) for t in tasks] + [12])
    cw = max([len(_label(k)) for k in keys] + [9]) + 1

    title = suite + (f" / {domain}" if domain else "")
    print(f"\n{title} — arm comparison (merged by agent+tag)\n" + "=" * (w + cw * len(keys) + 2))
    print(f"{'task':<{w}} " + "".join(f"{_label(k):>{cw}}" for k in keys))
    print("-" * (w + cw * len(keys) + 2))
    for t in tasks:
        row = f"{t:<{w}} "
        for k in keys:
            raw = groups[k]["raw"].get(t) or []
            row += f"{mean(raw) * 100:>4.0f}% n={len(raw):<2}".rjust(cw) if raw else "—".rjust(cw)
        print(row)
    print("-" * (w + cw * len(keys) + 2))

    print("\nsummary")
    for k in keys:
        g = groups[k]
        raw = g["raw"]
        n = sum(len(v) for v in raw.values())
        c = _consistency(raw)
        overall = mean([s for v in raw.values() for s in v]) * 100 if n else 0.0
        cs = "n/a (<2 runs/task)" if c != c else f"{c:.3f}"
        print(f"  {_label(k):<{cw}} overall={overall:5.1f}%  consistency={cs:<18} runs={n}  files={len(g['files'])}")
        if k != base and base_raw:
            arm_raw = [s for v in raw.values() for s in v]
            if arm_raw and all(s in (0.0, 1.0) for s in arm_raw + base_raw):
                a = round(sum(arm_raw))
                b = len(arm_raw) - a
                c2 = round(sum(base_raw))
                d2 = len(base_raw) - c2
                p = _fisher(a, b, c2, d2)
                print(f"{'':<{cw}} vs {_label(base)}: fisher p={p:.3f}  ({a}/{len(arm_raw)} vs {c2}/{len(base_raw)} passes)")
    print()


if __name__ == "__main__":
    app()
