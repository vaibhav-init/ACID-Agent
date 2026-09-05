# ACID-Agent

A data-analysis agent where every unit of work is a **transaction**:
explore → decide → generate+execute → validate → commit-or-rollback.
Only validated work reaches the workspace; failed attempts are rolled back and leave zero trace.

Implementation of *"Agentic Transactions: Towards ACID-Compliant Agent Systems"* (Sun, Wang, Li).

## What it does

```
Task ──► supervisor (≤20 units)
           │ per unit:
           │   explore (read-only, bounded)
           │   extract decisions
           │   write & run the unit script
           │   VALIDATION GATE: execution + LLM reflection
           │     + confidence-divergence probes (Qwen3-0.6B, local)
           │   pass  → git commit
           │   fail  → rollback → retry with feedback (≤2)
           ▼
      final answer + full provenance (JSONL trace + Postgres)
```

The gate rejects code that contradicts the evidence gathered during exploration
(token-level log-probability comparison, with vs without evidence), not just code that crashes.

## Results

Same model, same data, A/B:

| | baseline | ACID |
|---|---|---|
| overall KramaBench (6 domains) | ~74% | ~84% |
| hardest failures | 0% | 0% → 67% |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[confidence]"       # core + local scorer
docker compose up -d                 # Postgres + pgvector
python scripts/setup_confidence.py   # download Qwen3-0.6B
opencode auth login                  # backbone (Qwen family); or `claude login`
```

## Run

```bash
# one task
python scripts/acid_cli.py run-task "<question>" --agent claude-acid --data-dir path/to/csvs

# A/B evaluation
python scripts/acid_cli.py eval --agent claude-acid --domain biomedical --runs 3
python scripts/acid_cli.py eval --agent claude     --domain biomedical --runs 3
```

`claude-acid` = full transaction machinery. `claude` = same model, one raw session, no machinery.
Scores and consistency (`sqrt(mean per-task variance)`) land in `results/*.json`.

## Layout

- `acid_agent/graphs/` — supervisor + transaction unit (LangGraph)
- `acid_agent/validation.py` — the gate
- `acid_agent/confidence.py` — log-probability probes
- `acid_agent/workspace.py` — git-backed atomicity (rollback)
- `acid_agent/eval/` — KramaBench tasks + graders
- `db/schema.sql` — runs / units / validations / events
