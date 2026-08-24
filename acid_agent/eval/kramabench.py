"""Built-in evaluation suite (KramaBench-style data-to-insight tasks).

Three deterministic synthetic tasks with ground-truth graders:
  revenue_region : simple filter+aggregate
  mixed_dates    : TRAP — two date formats; naive parsing silently corrupts the answer
  top_product    : groupby + argmax, answer is a name

evaluate() runs each task n_runs times and reports mean score plus the paper's
consistency metric: sqrt(mean over tasks of per-task variance across runs).
Real KramaBench hookup: swap builtin_tasks() for loaders from the official repo.
"""

import json
import re
import time
from dataclasses import dataclass, field
from statistics import mean, variance
from typing import Callable

import numpy as np

NL = chr(10)


@dataclass
class Task:
    id: str
    question: str
    seed_files: dict[str, str] = field(default_factory=dict)
    grader: Callable[[str], float] = lambda answer: 0.0


def _first_number(text: str) -> float | None:
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m[0]) if m else None


def _num_grader(expected: float, tol_rel: float = 0.02) -> Callable[[str], float]:
    def grade(answer: str) -> float:
        val = _first_number(answer)
        if val is None:
            return 0.0
        return 1.0 if abs(val - expected) <= tol_rel * max(1.0, abs(expected)) else 0.0

    return grade


def _name_grader(expected: str) -> Callable[[str], float]:
    def grade(answer: str) -> float:
        return 1.0 if expected.lower() in answer.lower() else 0.0

    return grade


def _make_revenue_task() -> Task:
    rng = np.random.default_rng(42)
    regions = ["north", "south", "east", "west"]
    products = ["widget", "gadget", "doohickey"]
    lines = ["order_id,region,product,qty,unit_price"]
    expected_north = 0.0
    for i in range(200):
        region = regions[rng.integers(4)]
        product = products[rng.integers(3)]
        qty = int(rng.integers(1, 20))
        price = round(float(rng.uniform(5, 50)), 2)
        lines.append(f"{i},{region},{product},{qty},{price}")
        if region == "north":
            expected_north += qty * price
    expected = round(expected_north, 2)
    return Task(
        id="revenue_region",
        question=(
            "orders.csv lists orders with region, product, qty and unit_price. "
            f"What is the TOTAL revenue (qty * unit_price summed) for the region 'north'? "
            f"Answer with the number rounded to 2 decimals."
        ),
        seed_files={"orders.csv": NL.join(lines)},
        grader=_num_grader(expected),
    )


def _make_mixed_dates_task() -> Task:
    rng = np.random.default_rng(7)
    temps_jan = list(rng.normal(5, 3, 60))
    temps_feb = list(rng.normal(7, 3, 40))
    all_temps = temps_jan + temps_feb
    lines = ["date,temperature_c"]
    for i, t in enumerate(all_temps):
        day = (i % 28) + 1
        month = 1 if i < 60 else 2
        iso = f"2024-{month:02d}-{day:02d}"
        alt = f"2024/{month:02d}/{day:02d}"
        # TRAP: ~30% of dates use slashes instead of dashes
        lines.append(f"{alt if i % 3 == 0 else iso},{round(float(t), 1)}")
    expected = round(float(np.mean(all_temps)), 2)
    return Task(
        id="mixed_dates",
        question=(
            "readings.csv has a date column in MIXED formats (some use '-', some use '/'). "
            "What is the average temperature_c across ALL rows? Round to 2 decimals. "
            "Be careful: naive date parsing will silently drop or mis-bucket rows."
        ),
        seed_files={"readings.csv": NL.join(lines)},
        grader=_num_grader(expected),
    )


def _make_top_product_task() -> Task:
    rng = np.random.default_rng(99)
    products = ["alpha", "beta", "gamma", "delta"]
    rev = {p: 0.0 for p in products}
    lines = ["sale_id,product,amount"]
    for i in range(150):
        p = products[rng.integers(4)]
        amount = round(float(rng.uniform(10, 500)), 2)
        # make 'gamma' clearly the winner
        if p == "gamma":
            amount *= 3
        rev[p] += amount
        lines.append(f"{i},{p},{amount}")
    winner = max(rev, key=lambda p: rev[p])
    return Task(
        id="top_product",
        question=(
            "sales.csv has product and amount columns. Which product has the HIGHEST "
            "total revenue? Answer with just the product name."
        ),
        seed_files={"sales.csv": NL.join(lines)},
        grader=_name_grader(winner),
    )


def builtin_tasks() -> list[Task]:
    return [_make_revenue_task(), _make_mixed_dates_task(), _make_top_product_task()]


def evaluate(agent_name: str, runner_fn: Callable[[Task], str], n_runs: int = 3, out_dir="results"):
    """runner_fn(task) -> final answer string."""
    tasks = builtin_tasks()
    scores: dict[str, list[float]] = {t.id: [] for t in tasks}
    for t in tasks:
        for run in range(n_runs):
            try:
                answer = runner_fn(t)
            except Exception as e:
                print(f"[{agent_name}] {t.id} run {run}: ERROR {e}")
                answer = ""
            sc = t.grader(str(answer))
            scores[t.id].append(sc)
            print(f"[{agent_name}] {t.id} run {run}: score={sc}")

    per_task_mean = {k: mean(v) for k, v in scores.items()}
    all_runs = [s for v in scores.values() for s in v]
    overall = mean(all_runs) if all_runs else 0.0
    variances = [variance(v) for v in scores.values() if len(v) > 1]
    consistency = (sum(variances) / len(variances)) ** 0.5 if variances else 0.0

    report = {
        "agent": agent_name,
        "n_runs": n_runs,
        "per_task_mean": per_task_mean,
        "overall_score": round(overall * 100, 1),
        "consistency_sqrt_avg_var": round(consistency, 3),
        "raw_scores": scores,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = __import__("pathlib").Path(out_dir)
    out.mkdir(exist_ok=True)
    path = out / f"eval_{agent_name}_{int(time.time())}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("agent", "overall_score", "consistency_sqrt_avg_var")}, indent=2))
    print(f"saved -> {path}")
    return report