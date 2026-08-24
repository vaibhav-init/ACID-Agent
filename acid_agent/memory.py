"""Transaction-aware memory graph in Postgres (semantic durability).

Nodes/edges persist across runs. After each COMMITTED unit a "memory LLM" decides
evolution ops (insert / merge / delete / split) so the graph stays compact and useful.
Only validated units ever touch this graph — failed attempts are isolated away.
"""

import json

from .config import get_conn
from .llm import ask_structured
from .schemas import MemoryOp, MemoryOps


def remember(key: str, content: str, related_keys: list[str] | None = None, source_run=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO memory_nodes (key, content, source_run) VALUES (%s, %s, %s)
               ON CONFLICT (key) DO UPDATE SET content = EXCLUDED.content,
                   updated_at = now()""",
            (key, content, source_run),
        )
        for rel in related_keys or []:
            conn.execute(
                """INSERT INTO memory_edges (src_key, dst_key, rel) VALUES (%s, %s, 'related')
                   ON CONFLICT DO NOTHING""",
                (key, rel),
            )


def forget(key: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM memory_nodes WHERE key = %s", (key,))


def merge(src_key: str, dst_key: str):
    """Fold src into dst, then delete src."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content FROM memory_nodes WHERE key = %s", (src_key,)
        ).fetchone()
        if not row:
            return
        conn.execute(
            """UPDATE memory_nodes SET content = content || chr(10) || %s,
                   updated_at = now() WHERE key = %s""",
            (row[0], dst_key),
        )
        conn.execute("DELETE FROM memory_nodes WHERE key = %s", (src_key,))


def all_keys(limit: int = 60) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT key FROM memory_nodes ORDER BY updated_at DESC LIMIT %s", (limit,)
        ).fetchall()
    return [r[0] for r in rows]


def context_block(max_chars: int = 3000) -> str:
    """Render recent memory as a text block for agent prompts."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT key, content FROM memory_nodes ORDER BY updated_at DESC LIMIT 25"
        ).fetchall()
    lines = [f"- {k}: {c}" for k, c in rows]
    return chr(10).join(lines)[:max_chars]


OPS_PROMPT = """You maintain the long-term memory of a data-analysis agent as a knowledge graph.

Task just completed (validated unit summary):
{summary}

Existing memory keys:
{keys}

Decide evolution operations. Rules:
- insert: new durable fact/lesson worth keeping (short, self-contained).
- merge: fold new info into an existing key instead of creating a near-duplicate.
- delete: remove a key that is wrong or obsolete.
Prefer merging over inserting duplicates. Max 4 ops. Return [] if nothing is worth remembering."""


def evolve_from_unit(summary: str, run_id=None, tracer=None) -> list[MemoryOp]:
    """Ask the memory LLM for evolution ops after a committed unit, then apply them."""
    keys = all_keys()
    result: MemoryOps = ask_structured(
        OPS_PROMPT.format(summary=summary[:2000], keys=keys), MemoryOps
    )
    ops = result.ops
    for op in ops:
        if op.op == "insert":
            remember(op.key, op.content, op.related_keys, source_run=run_id)
        elif op.op == "merge":
            remember(op.key, op.content, source_run=run_id)
            for other in op.related_keys or []:
                merge(other, op.key)
        elif op.op == "delete":
            forget(op.key)
        elif op.op == "split":  # treat as insert of the new fragment
            remember(op.key, op.content, source_run=run_id)
        if tracer:
            tracer.log("memory_op", op=json.dumps(op.model_dump()))
    return ops