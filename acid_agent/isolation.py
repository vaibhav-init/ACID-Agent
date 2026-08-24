"""Agent-level isolation (semantic isolation).

Three dependency-aware policies for spawning sub-agents:

  independent   — sub-agents on disjoint sub-tasks run fully parallel, each in its
                  own cloned workspace; results are collected independently.
  collaborative — sub-agents build one shared artifact: each works on its own git
                  branch of the same workspace, then branches merge sequentially.
  competitive   — N agents attack the SAME task in isolated workspace clones;
                  the best trajectory wins (most committed units, then longest output).

Isolation here = separate workspaces/branches so failed or conflicting agents
cannot contaminate shared state. (Docker-level sandboxing is a drop-in hardening:
swap _clone_workspace for a container mount.)
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .workspace import Workspace


def _clone_workspace(src: Workspace, base_dir: Path, slug: str) -> Workspace:
    """Copy-on-write style clone: fresh dir with the committed state copied over."""
    dst_root = base_dir / slug
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(
        src.root,
        dst_root,
        ignore=shutil.ignore_patterns(".git"),
    )
    # re-init git history so clones can commit independently
    from git import Repo

    repo = Repo.init(dst_root)
    repo.git.add(A=True)
    repo.git.commit(m="clone: isolated copy")
    return Workspace(dst_root)


def spawn_independent(parent_ws: Workspace, subtasks: list[dict], runner_fn, base_dir="workspaces"):
    """subtasks: [{slug, seed_files, prompt}]; runner_fn(ws, subtask) -> result dict."""
    clones = [
        (_clone_workspace(parent_ws, Path(base_dir), st["slug"]), st) for st in subtasks
    ]
    with ThreadPoolExecutor(max_workers=min(4, len(clones))) as pool:
        futures = [pool.submit(runner_fn, ws_clone, st) for ws_clone, st in clones]
        results = [f.result() for f in futures]
    return results


def spawn_collaborative(parent_ws: Workspace, subtasks: list[dict], runner_fn):
    """Each agent commits to its own branch; branches merge into main in order."""
    branch_results = []
    main_head = parent_ws.head()
    for i, st in enumerate(subtasks):
        branch = f"agent-{i}-{st['slug']}"
        parent_ws.repo.git.checkout("-b", branch)
        try:
            res = runner_fn(parent_ws, st)
            parent_ws.repo.git.add(A=True)
            dirty = parent_ws.repo.is_dirty(untracked_files=True)
            if dirty:
                parent_ws.repo.git.commit(m=f"{branch}: {st['slug']}")
            branch_results.append({"branch": branch, "committed": dirty, "result": res})
        except Exception as e:
            branch_results.append({"branch": branch, "committed": False, "error": str(e)})
        finally:
            parent_ws.repo.git.checkout("main")
    # sequential merge of validated branch work into main
    for br in branch_results:
        if br["committed"]:
            try:
                parent_ws.repo.git.merge(br["branch"], "--no-edit")
            except Exception:
                parent_ws.repo.git.merge("--abort")
    _ = main_head
    return branch_results


def spawn_competitive(task: str, n: int, runner_fn, base_dir="workspaces", slug_prefix="compete"):
    """N independent full attempts on cloned workspaces; best trajectory wins.

    runner_fn(ws, {"task": task}) -> {"score": float, ...}
    """
    clones = []
    for i in range(n):
        root = Path(base_dir) / f"{slug_prefix}_{i}"
        root.mkdir(parents=True, exist_ok=True)
        clones.append(Workspace.create(base_dir, f"{slug_prefix}_{i}"))
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(runner_fn, ws, {"task": task}) for ws in clones]
        results = [f.result() for f in futures]
    best_idx = max(range(n), key=lambda i: results[i].get("score", 0))
    return {"best_index": best_idx, "best": results[best_idx], "all": results}