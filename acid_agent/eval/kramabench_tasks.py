"""KramaBench task loader — loads real KramaBench tasks from official workload JSON files.

Each task has:
  - id: unique identifier (e.g., "archeology-hard-1")
  - query: natural language question
  - answer: expected answer
  - answer_type: numeric_exact, list_exact, etc.
  - data_sources: list of data files needed
  - subtasks: optional sub-questions with their own answers
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


KRAMABENCH_DIR = Path(__file__).resolve().parents[2] / "vendor" / "acid-paper-ref" / "Kramabench"
KRAMABENCH_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "kramabench" / "data"

DOMAINS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]


@dataclass
class KramaTask:
    """A single KramaBench task with its expected answer and data files."""
    id: str
    domain: str
    question: str
    answer: object  # expected answer (number, string, or list)
    answer_type: str  # numeric_exact, list_exact, etc.
    data_sources: list[str] = field(default_factory=list)
    subtasks: list[dict] = field(default_factory=list)
    data_dir: Path | None = None  # resolved path to input directory

    @property
    def seed_files(self) -> dict[str, bytes]:
        """Read data files as {relative_name: bytes} for workspace seeding.

        Bytes, not text: the workload mixes CSVs with binary formats (.xlsx);
        a text round-trip would silently corrupt those files.

        Resolution is deliberately forgiving, because a miss here produces an
        EMPTY workspace and a 0.00 score that looks like an agent failure:
          1. exact relative path under input/
          2. glob patterns ("State MSA Identity Theft Data/*")
          3. recursive search by basename (legal/ nests its files in
             subdirectories, so the exact path never matches)
        """
        files: dict[str, bytes] = {}
        if not self.data_dir:
            return files
        for src in self.data_sources:
            fpath = self.data_dir / src
            if fpath.is_file():
                files[src] = fpath.read_bytes()
                continue
            if any(ch in src for ch in "*?["):
                for m in sorted(self.data_dir.glob(src)):
                    if m.is_file():
                        files[m.relative_to(self.data_dir).as_posix()] = m.read_bytes()
                continue
            if fpath.is_dir():
                for m in sorted(fpath.rglob("*")):
                    if m.is_file() and not m.name.startswith("."):
                        files[m.relative_to(self.data_dir).as_posix()] = m.read_bytes()
                continue
            # Fall back to basename search; take the shallowest match so a
            # top-level file wins over a copy buried in an archive subtree.
            hits = [m for m in self.data_dir.rglob(Path(src).name) if m.is_file()]
            if hits:
                best = min(hits, key=lambda m: len(m.relative_to(self.data_dir).parts))
                files[Path(src).name] = best.read_bytes()
        return files


def load_domain(domain: str) -> list[KramaTask]:
    """Load all tasks for a domain from the official workload JSON."""
    workload_path = KRAMABENCH_DIR / "workload" / f"{domain}.json"
    if not workload_path.exists():
        raise FileNotFoundError(f"Workload not found: {workload_path}")

    with open(workload_path) as f:
        tasks_raw = json.load(f)

    data_dir = KRAMABENCH_DATA_DIR / domain / "input"
    if not data_dir.exists():
        data_dir = None  # data not downloaded yet

    tasks = []
    for t in tasks_raw:
        tasks.append(KramaTask(
            id=t["id"],
            domain=domain,
            question=t["query"],
            answer=t.get("answer"),
            answer_type=t.get("answer_type", "numeric_exact"),
            data_sources=t.get("data_sources", []),
            subtasks=t.get("subtasks", []),
            data_dir=data_dir,
        ))
    return tasks


def load_all_domains() -> dict[str, list[KramaTask]]:
    """Load tasks for all domains."""
    return {d: load_domain(d) for d in DOMAINS}


def get_available_domains() -> list[str]:
    """Return domains that have both workload JSON and data files."""
    available = []
    for d in DOMAINS:
        workload = KRAMABENCH_DIR / "workload" / f"{d}.json"
        data = KRAMABENCH_DATA_DIR / d / "input"
        if workload.exists() and data.exists():
            available.append(d)
    return available


# --- Graders (from KramaBench evaluate.py) ---

def _ascii_minus(s: str) -> str:
    """Normalize typographic minus signs to ASCII before number extraction.

    Models format negatives with U+2212 MINUS SIGN. The number regex only
    accepts "-", so "\u22120.008004" was captured as +0.008004 and a
    six-decimal-correct answer scored 0.0. Only unambiguous minus characters are
    converted -- en/em dashes are left alone because they usually mark ranges.
    """
    return s.replace("\u2212", "-").replace("\u2796", "-")


def grade_numeric(expected: float, answer: str, tol_rel: float = 0.005) -> float:
    """Grade a numeric answer with relative tolerance (KramaBench uses 0.005).

    Scans every number in the answer and passes if ANY is within tolerance —
    final answers often embed the value in a sentence ("The average is 3.1333"),
    so taking the first or last number alone is fragile.
    """
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", _ascii_minus(answer).replace(",", ""))
    for tok in nums:
        try:
            val = float(tok)
        except ValueError:
            continue
        if abs(val - expected) <= tol_rel * max(1.0, abs(expected)):
            return 1.0
    return 0.0


def _norm(s: str) -> str:
    """Casefold + strip diacritics for string matching.

    Agents routinely drop diacritics (ASCII "Sao Paulo" for the accented form);
    a raw substring match scores that 0.0, which is a grading artifact rather
    than a wrong answer.
    """
    import unicodedata
    d = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in d if not unicodedata.combining(c)).casefold()


def grade_exact(expected: object, answer: str) -> float:
    """Grade exact match (string or numeric)."""
    if isinstance(expected, (int, float)):
        return grade_numeric(float(expected), answer)
    if isinstance(expected, str):
        return 1.0 if _norm(expected) in _norm(answer) else 0.0
    if isinstance(expected, list):
        # Partial credit: fraction of expected items present in the answer
        na = _norm(answer)
        matches = sum(1 for item in expected if _norm(item) in na)
        return matches / len(expected) if expected else 0.0
    return 0.0


# Answer types these graders cannot judge faithfully. Their expected values are
# prose ("The average period is 11 years, with maxima in 1968, 1979, ...") or
# fuzzy lists, and substring matching scores a correct answer 0.0 — a fake floor
# indistinguishable from agent failure. KramaBench judges these separately; until
# that judge is wired, they are excluded rather than silently mis-scored.
UNGRADEABLE_TYPES = {"string_approximate", "list_approximate"}

# "approximate" numerics are still numbers, just not to 0.005.
APPROX_NUMERIC_TOL = 0.05


def is_gradeable(task: KramaTask) -> bool:
    return task.answer is not None and task.answer_type not in UNGRADEABLE_TYPES


def grade_task(task: KramaTask, answer: str) -> float | None:
    """Score in [0,1], or None when the answer type cannot be judged faithfully.

    None is not zero: callers must exclude it from aggregates instead of counting
    it as a failure.
    """
    if task.answer is None:
        return 0.0
    if task.answer_type in UNGRADEABLE_TYPES:
        return None
    if task.answer_type == "numeric_exact":
        return grade_numeric(float(task.answer), answer)
    if task.answer_type == "numeric_scientific_exact":
        return grade_numeric(float(task.answer), answer)
    if task.answer_type == "numeric_approximate":
        return grade_numeric(float(task.answer), answer, tol_rel=APPROX_NUMERIC_TOL)
    return grade_exact(task.answer, answer)
