# ACID-Agent

Implementation of **"Agentic Transactions: Towards ACID-Compliant Agent Systems"** (Sun, Wang, Li — Tsinghua).
A data-analysis agent where every unit of work is a _semantic transaction_:
**explore → decide → generate+execute → validate → commit-or-rollback.**

Only validated work reaches the workspace (git) and memory (Postgres knowledge graph).
Failed attempts leave zero trace.

## Architecture in one picture

```
Task ──► skill router ──► supervisor graph (≤20 units)
                            │ per unit:
                            │  explore (read-only, redundancy-bounded)
                            │  extract decisions
                            │  OpenCode writes & runs unit script
                            │  VALIDATION GATE: execution + decision divergence
                            │    + code-span divergence + LLM reflection
                            │  pass → git commit + memory evolution
                            │  fail → rollback → retry (≤2) with feedback
                            ▼
                       final answer + full provenance
```

Confidence = exp(mean token logprob) from a local Qwen3-0.6B (APIs hide logprobs).
Divergence = confidence of the same text _with_ vs _without_ evidence.

## Setup

```bash
# 1. Python env (Python 3.11+; repo developed on 3.14)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core agent
pip install -e ".[confidence]"   # optional: local Qwen3-0.6B scorer (~2 GB torch)

# 2. Config
cp .env.example .env             # then edit: LLM_PROVIDER, key, LLM_MODEL

# 3. Postgres (+pgvector)
docker compose up -d             # schema auto-loads on first start

# 4. Local confidence model (optional but recommended)
python scripts/setup_confidence.py

# 5. OpenCode auth (the code-execution backbone)
opencode auth login              # pick the same provider as .env
```

## Run one task

```bash
python scripts/acid_cli.py run-task "What is the total revenue for region north?" \
    --agent acid --data-dir path/to/csvs

python scripts/acid_cli.py run-task "..." --agent baseline   # ablation control (no ACID)
```

## Evaluate (score + consistency)

```bash
python scripts/acid_cli.py eval --agent acid --runs 3
python scripts/acid_cli.py eval --agent baseline --runs 3
```

Reports `overall_score` and the paper's consistency metric
`sqrt(mean per-task variance across runs)` into `results/*.json`.

Built-in suite (`acid_agent/eval/kramabench.py`) ships three deterministic tasks,
including a mixed-date-format trap that silently corrupts naive agents.
To use the real KramaBench: replace `builtin_tasks()` with loaders from the
official repo — the Task dataclass (question + seed files + grader) is the contract.

## Tests

```bash
pytest -q            # workspace/confidence/graph-wiring tests (graph tests need DB up)
pytest skills -q     # skill validation suites
```

## Where each ACID property lives

| Property    | Code                                                                                  |
| ----------- | ------------------------------------------------------------------------------------- |
| Atomicity   | `workspace.py` (snapshot→commit-or-revert), `graphs/unit_graph.py`                    |
| Consistency | `confidence.py`, `validation.py` (gate thresholds 0.25 / 0.50 / 0.45)                 |
| Isolation   | `isolation.py` (independent / collaborative branches / competitive clones)            |
| Durability  | `db/schema.sql`, `memory.py`, `tracer.py` (append-only events), LangGraph checkpoints |

## Notes & honest limitations

- Competitive isolation currently uses process-level clones; Docker sandboxing is a drop-in upgrade.
- Skill embeddings column exists (pgvector) but routing is LLM-choice based; wire embeddings when an embedding API is available.
- Real KramaBench hookup is documented above but not bundled (licensing/download).
