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
                            │  Claude Code writes & runs unit script
                            │  VALIDATION GATE: execution + decision divergence
                            │    + code-span divergence + LLM reflection
                            │  pass → git commit + memory evolution
                            │  fail → rollback → retry (≤2) with feedback
                            ▼
                       final answer + full provenance
```

Confidence = exp(mean token logprob) from a local Qwen3-0.6B (running on CUDA).
Divergence = confidence of the same text _with_ vs _without_ evidence.

## Setup

```bash
# 1. Python env (Python 3.11+; repo developed on 3.14)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core agent
pip install -e ".[confidence]"   # optional: local Qwen3-0.6B scorer (~2 GB torch)

# 2. Config
cp .env.example .env             # thresholds & DB url; no API keys needed

# 3. Postgres (+pgvector)
docker compose up -d             # schema auto-loads on first start

# 4. Local confidence model (optional but recommended)
python scripts/setup_confidence.py

# 5. Claude Code CLI (the code-execution backbone for BOTH agent arms)
npm install -g @anthropic-ai/claude-code   # v2.x
claude login                               # subscription auth; the agent inherits it
# Optional: set CLAUDE_MODEL in .env to pin one model across both arms for reproducible evals.
```

## Run one task

```bash
python scripts/acid_cli.py run-task "What is the total revenue for region north?" \
    --agent claude-acid --data-dir path/to/csvs

python scripts/acid_cli.py run-task "..." --agent claude     # raw harness baseline (no ACID)
python scripts/acid_cli.py run-task "..." --agent claude-acid # (acid is an alias)
```

## Evaluate (score + consistency)

```bash
python scripts/acid_cli.py eval --agent claude-acid --domain archeology --runs 3
python scripts/acid_cli.py eval --agent claude    --domain archeology --runs 3
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

## Tracing (LangSmith)

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=acid-agent
```

`acid_agent/tracing.py` exports these to `os.environ` at package import — setting them
in `.env` alone is not enough, since nothing else in the repo reads `.env` into the
environment. LangGraph nodes then trace automatically; the backbone `claude -p`
subprocess calls are annotated explicitly (`backbone.ask`, `backbone.ask_structured`,
`claude_code.session`, `validation_gate`) because LangChain callbacks can't see them.
Seeded data files are stripped from trace inputs. With tracing off, `@traced` is inert.

Skills for querying traces live in `.claude/skills/langsmith-*`
(from [langsmith-skills](https://github.com/langchain-ai/langsmith-skills)); the
`langsmith` CLI they use is an optional separate install.

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
