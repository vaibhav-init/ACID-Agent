"""Git-backed transactional workspace (atomicity + isolation backbone).

Lifecycle per task:
  create()            -> dir + git init + initial commit containing all input data
  head()              -> current commit sha
  rollback()          -> reset --hard + clean  (failed attempt leaves zero trace)
  commit(msg)         -> add -A + commit, returns sha   (only called after validation passes)
  run_code(code)      -> write snippet.py + execute with timeout (used by gate & agents)
"""

import subprocess
import sys
from pathlib import Path

from git import Repo

from .schemas import ExecResult


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.repo = Repo(root)

    @classmethod
    def create(cls, base_dir: str | Path, task_slug: str, seed_files: dict[str, str] | None = None):
        root = Path(base_dir) / task_slug
        root.mkdir(parents=True, exist_ok=True)
        repo = Repo.init(root)
        for name, content in (seed_files or {}).items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        repo.git.add(A=True)
        # allow-empty keeps head() valid even with no seed files
        repo.git.commit(m="init: seed workspace", allow_empty=True)
        return cls(root)

    def head(self) -> str:
        return self.repo.head.commit.hexsha[:10]

    def rollback(self):
        """Discard ALL uncommitted changes — failed attempts leave zero trace."""
        self.repo.git.reset("--hard", "HEAD")
        self.repo.git.clean("-fd")

    def commit(self, msg: str) -> str:
        self.repo.git.add(A=True)
        if not self.repo.is_dirty(untracked_files=True):
            return self.head()
        self.repo.git.commit(m=msg)
        return self.head()

    def diff_summary(self) -> str:
        """Human-readable summary of uncommitted changes (for reflection/feedback)."""
        out = []
        for item in self.repo.index.diff("HEAD"):
            out.append(f"M {item.a_path}")
        for path in self.repo.untracked_files:
            out.append(f"+ {path}")
        return "; ".join(out) if out else "(no changes)"

    # ---- code execution ----

    def run_code(self, code: str, name: str = "snippet.py", timeout_s: int = 180) -> ExecResult:
        path = self.root / name
        path.write_text(code, encoding="utf-8")
        return self.run_script(name, timeout_s=timeout_s)

    def run_script(self, rel_path: str, timeout_s: int = 180) -> ExecResult:
        try:
            proc = subprocess.run(
                [sys.executable, rel_path],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return ExecResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout[-8000:],
                stderr=proc.stderr[-8000:],
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                ok=False,
                stdout=(e.stdout or b"").decode(errors="ignore")[-4000:] if e.stdout else "",
                stderr=f"TIMEOUT after {timeout_s}s",
                returncode=-1,
            )