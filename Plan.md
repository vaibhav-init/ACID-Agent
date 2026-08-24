# ACID-Agent — Implementation Plan (v2)

Blueprint for implementing **"Agentic Transactions: Towards ACID-Compliant Agent Systems"** (Sun, Wang, Li — Tsinghua University): an LLM data-agent harness where every unit of work is a _semantic transaction_ with commit-or-retry semantics, validated by confidence signals before anything touches persistent state.

**v2 stack decisions:** LangChain + LangGraph for orchestration · an open-source coding harness (**OpenCode**) as the code-generation/execution backbone · **PostgreSQL** as the system of record (checkpoints, event log, memory graph, skill registry).

---

## 1. What We're Building (Recap)

A data-science agent that executes tasks as a bounded sequence of **Atomic Semantic Transaction Units**:

```
Task ──► [Skill Router: validated skill exists?] ──yes──► invoke skill ──► done
              │no
              ▼
  UNIT i (of ≤20):
  ┌────────────────────────────────────────────────────┐
  │ 1. EXPLORATION (read-only exec)                    │
  │    budget: 1–4 rounds (-1 every 2 units)           │
  │    observations → evidences → consolidated summary │
  │    redundancy check: divergence > 0.45 ⇒ stop      │
  │ 2. DECISION EXTRACTION                             │
  │    structured decisions extracted from evidence    │
  │ 3. CODE GENERATION + EXECUTION  ◄── OpenCode       │
  │    (read-only phase, then append-only phase)       │
  │ 4. VALIDATION GATE                                 │
  │    • execution error?                              │
  │    • decision divergence < 0.25 ⇒ retry            │
  │    • max code-span divergence < 0.50 ⇒ retry       │
  │    • LLM reflection feedback                       │
  │    FAIL ⇒ retry w/ feedback (max 2/unit);          │
  │           failed attempt fully isolated            │
  │ 5. COMMIT                                          │
  │    append-only workspace write + git commit        │
  │    memory-graph update in Postgres (validated only)│
  └────────────────────────────────────────────────────┘
```

Only **validated** effects reach the workspace and memory. Failed retries leave zero trace (isolation). Everything committed is journaled with provenance (durability).

---

## 2. ARCHITECTURE OPTIONS

### Option A — Custom Lightweight Harness

Pure-Python ReAct loop (~2–3k LOC). Max control, most upfront work. _Rejected in v2 per your preference for LangChain._

### Option B — LangGraph-Based ✅ (CHOSEN)

Model the transaction unit as a state machine: `explore → decide → generate → validate → (retry edge | commit edge)`.

| Pros                                                                   | Cons                                                     |
| ---------------------------------------------------------------------- | -------------------------------------------------------- |
| Built-in checkpointing — **PostgresSaver** gives crash-safe durability | Checkpointing is step-level; staging/rollback still ours |
| Conditional edges = natural fit for commit-or-retry gates              | Graph indirection around the retry loop                  |
| `Send` API for parallel sub-agents; subgraphs for isolated contexts    | Version churn between releases                           |
| Native LangChain model/tool/prompt primitives + LangSmith tracing      |                                                          |

### Option C — smolagents (HuggingFace)

Tiny code-agent; no transactional machinery; would rewrite most of it anyway.

### Option D — Multi-Agent Frameworks (AutoGen/AG2, CrewAI, OpenAI Agents SDK)

Useful patterns for the isolation layer only; their runtimes fight staged execution and "failed agents leave no trace."

### Option E — Wrap an Existing Closed Harness (Claude Agent SDK)

Cannot intercept the internal loop to insert confidence gates.

> **Decision (v2):** **LangGraph orchestrates the ACID transaction loop** (Option B). Inside each unit, code generation + execution is delegated to an **open-source coding harness (OpenCode, headless)**. LangChain supplies the model/tool/prompt primitives used inside nodes. **PostgreSQL** holds all durable state.

### 2.1 How LangGraph + OpenCode Fit Together (integration pattern)

**Pattern 1 — LangGraph is the brain, OpenCode is the hands (recommended):**

```
LangGraph StateGraph (task level)
 └─ unit_node(i)  ──►  Unit StateGraph (transaction level)
                        ├─ explore_node        → OpenCode `run` (read-only prompt) or direct LLM
                        ├─ extract_decisions   → LangChain structured-output chain
                        ├─ generate_execute    → OpenCode headless (`opencode run`) in workspace
                        ├─ validate_gate       → confidence engine + reflection judge
                        │     ├─ fail → conditional edge back (retry ≤2, feedback injected)
                        │     └─ pass → commit_node
                        └─ commit_node         → git commit + Postgres memory update
```

- OpenCode runs **non-interactively** (`opencode run "<instruction>"`) pointed at the task workspace; we parse its emitted diffs/output.
- **Read-only vs append-only** enforcement: snapshot git state before the call → allow OpenCode to work → if validation fails, `git checkout .` restores pre-call state (failed attempt leaves zero trace). If it passes, the diff is committed.
- OpenCode's provider/model is configurable, so the backbone LLM stays swappable.

**Pattern 2 — OpenCode as the whole harness, LangGraph only wrapping retries/validation:** simpler but we lose control of internal steps (weaker exploration budgeting, no clean read-only phase). Not recommended.

### 2.2 Where LangGraph Is Used (concrete answers to "can we use LangGraph anywhere?")

Yes — it becomes the backbone in six places:

1. **Transaction-unit graph**: `explore → decide → generate → validate → commit` with a bounded conditional retry edge (≤2) — this IS the paper's commit-or-retry semantics expressed natively.
2. **Task-level supervisor graph**: unit counter (≤20), exploration-budget decay (-1 round every 2 units), skill-router entry node, final answer assembly.
3. **Durability**: `AsyncPostgresSaver` checkpointer persists every super-step to Postgres → crash recovery + replay for free.
4. **Isolation**: `Send` API spawns parallel **Independent** sub-agents; **Collaborative** agents = subgraphs sharing a state channel with periodic sync; **Competitive** agents = separate graph invocations in Docker containers, best trajectory wins.
5. **Human-in-the-loop (optional)**: `interrupt()` at the validation gate for manual approval mode.
6. **Observability**: first-class LangSmith tracing; Langfuse also pluggable.

### Architectural Style Choices (orthogonal)

| Dimension                 | Options                                                     | Recommendation                                                   |
| ------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------- |
| Process model             | Single asyncio process / multiprocessing / worker-per-agent | **asyncio single process**; OpenCode invoked as subprocess       |
| Validation gate placement | In-graph node / sidecar judge service                       | **In-graph node** calling local confidence model over HTTP       |
| State ownership           | In-memory objects / DB as source of truth                   | **Postgres as source of truth**; LangGraph state is a view       |
| Sub-agent transport       | Direct asyncio / message queue (Redis/NATS)                 | **Direct asyncio + Send API**; queue only if we scale out        |
| Workspace layout          | One repo per task / shared repo with branches               | **One git repo per task**, branch per sub-agent, merge on commit |

---

## 3. TECH STACK OPTIONS (per layer)

### 3.0 Orchestration Framework

| Option                       | Verdict                                            |
| ---------------------------- | -------------------------------------------------- |
| **LangChain + LangGraph** ✅ | Chosen — primitives + state-machine agent runtime  |
| LlamaIndex Workflows         | Event-driven alternative; weaker checkpointing     |
| PydanticAI                   | Typed agents; smaller ecosystem                    |
| Custom pure-Python           | Max control (v1 recommendation); more upfront work |

Division of labor: **LangChain** = chat models, prompts, structured output, tools, retrievers. **LangGraph** = the transactional state machine, checkpointing, parallelism, interrupts.

### 3.1 Language

| Option              | Verdict                                                          |
| ------------------- | ---------------------------------------------------------------- |
| **Python 3.11+** ✅ | LangChain/LangGraph native; `ast` for code analysis; data stack  |
| TypeScript/Node     | OpenCode itself is TS, but our orchestrator benefits from Python |
| Rust/Go             | Overkill                                                         |

### 3.2 Coding Harness / Execution Backbone (open-source)

| Option           | Notes                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **OpenCode** ✅  | Open-source terminal coding agent (sst/opencode); multi-provider; **headless `opencode run`**; client/server + SDK; LSP-aware |
| Aider            | Git-native pair-programming CLI; strong edit formats; less agentic                                                            |
| Goose (Block)    | Extensible open-source agent; headless mode                                                                                   |
| OpenHands        | Full platform (sandbox, browser); heavier                                                                                     |
| Cline / Roo Code | VS Code-centric; harder to drive headless                                                                                     |

> ⚠️ **"freebuff"** — I could not identify an open-source agent harness by this name. Closest known projects are listed above; please confirm which you meant (see §8 Open Decisions).

Integration detail: we drive OpenCode **headlessly per step** with tightly scoped instructions ("explore these files read-only", "implement decision D as code"), capture stdout + diffs, and enforce staging via git snapshots around each call.

### 3.3 Backbone LLM (reasoning model behind OpenCode + LangChain nodes)

| Option                              | Notes                                     |
| ----------------------------------- | ----------------------------------------- |
| **DashScope/Bailian (Qwen family)** | Paper's setup; OpenAI-compatible          |
| GLM (Zhipu) via API                 | Paper's second backbone                   |
| OpenRouter / Together / Fireworks   | Easy swapping across many open models     |
| Anthropic / OpenAI direct           | Strongest models; watch cost on retries   |
| Self-hosted via vLLM/SGLang         | Full control + native logprobs; needs GPU |

All consumed through the **`openai` SDK** (or OpenCode's provider config) — swappable behind one interface.

### 3.4 Confidence Estimation (critical component)

| Option                                                        | Trade-offs                                               |
| ------------------------------------------------------------- | -------------------------------------------------------- |
| **Local Qwen3-0.6B via vLLM** ✅ (paper's choice)             | Cheap, fast, real token logprobs; ~2 GB VRAM or CPU-only |
| HuggingFace `transformers` direct                             | Simpler deps, slower batch scoring                       |
| API `logprobs` (OpenAI/vLLM endpoints)                        | Zero extra infra IF provider exposes top_logprobs        |
| Self-consistency sampling (N samples, agreement = confidence) | Model-agnostic but N× token cost                         |
| Verbalized confidence ("rate 0–1")                            | Cheapest, least reliable                                 |
| P(True) / semantic entropy methods                            | Research-grade calibration, more complexity              |

> Still required even with OpenCode: harness providers rarely expose token logprobs, and the paper explicitly uses a local small model for scoring.

### 3.5 Code Execution Path

| Option                                     | Verdict                                                  |
| ------------------------------------------ | -------------------------------------------------------- |
| **OpenCode's built-in bash/file tools** ✅ | Default exec path; matches "agent manipulates workspace" |
| Jupyter kernel (`ipykernel`) side-session  | Optional for stateful multi-step data sessions           |
| Docker container wrapping OpenCode         | ✅ for Competitive isolation + untrusted tasks           |
| gVisor / Firecracker                       | Later hardening                                          |
| E2B / Modal (cloud sandboxes)              | If no local Docker                                       |

Read-only vs append-only phases are enforced by **git snapshot → run → validate → commit-or-revert**, not by trusting the harness.

### 3.6 Database — PostgreSQL ✅ (system of record)

| Role in our system                          | Implementation                                                                                            |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| LangGraph checkpoints (crash safety/replay) | `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`)                                                    |
| Append-only event log / WAL + provenance    | `events` table (JSONB payloads, monotonic seq) — every LLM call, tool call, artifact hash                 |
| Transaction-aware memory graph              | `memory_nodes` / `memory_edges` tables (JSONB props); optional **Apache AGE** extension for graph queries |
| Skill registry + routing                    | `skills` table + **pgvector** embeddings for semantic retrieval                                           |
| Run/task metadata & metrics                 | `runs`, `units`, `validations`, `scores` tables                                                           |

Alternatives considered: SQLite (fine locally, weaker concurrency/extensions), DuckDB (analytics, not OLTP), Neo4j (dedicated graph DB — overkill when Postgres + JSONB/AGE suffices).

### 3.7 Memory / Knowledge Graph (durability)

| Option                                     | Verdict                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| **Postgres tables (nodes/edges JSONB)** ✅ | Transaction-aware evolution ops with SQL auditability; one DB for everything |
| NetworkX in-process cache over Postgres    | Optional convenience layer for traversal                                     |
| Neo4j                                      | Heavier infra; revisit if graph queries dominate                             |
| Mem0 / Zep-Graphiti / Letta / MemOS        | Off-the-shelf memory; less control over transaction-aware evolution          |

Evolution ops (insert/merge/split/delete) performed by a dedicated "memory LLM" after each committed unit; last **15 units** kept in active context.

### 3.8 Multi-Agent Isolation

| Policy        | Mechanism                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------- |
| Independent   | LangGraph `Send` API — parallel subgraph invocations, disjoint resource subsets                 |
| Collaborative | Branch-per-agent git workspaces + periodic context exchange via shared state channel + merge    |
| Competitive   | Separate graph runs inside Docker containers on cloned workspaces; early termination; best wins |

Policy selection: heuristic router first; fine-tuned policy predictor deferred (paper's future work).

### 3.9 Skill Hub (offline atomicity/consistency)

| Option                                                            | Verdict                                                                                                  |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Skills = folders + typer CLI + auto-generated pytest suite** ✅ | Paper-faithful ("repos packaged as skills with standardized CLIs, validated by generated test suites")   |
| Skills as MCP servers/tools                                       | Interop standard; adopt later for cross-harness reuse                                                    |
| Registry + routing                                                | Postgres `skills` table + **pgvector** embedding retrieval → LLM rerank; re-weighted from execution logs |

### 3.10 Observability & Tracing

| Option                              | Verdict                                             |
| ----------------------------------- | --------------------------------------------------- |
| **LangSmith** ✅ (native LangChain) | Zero-config tracing of every graph step/LLM call    |
| Langfuse (self-hosted)              | Alternative if we want self-hosted UI               |
| Postgres `events` table             | Our own WAL/provenance record (required regardless) |
| MLflow / Phoenix / OTel             | Optional later                                      |

### 3.11 Data Processing

**pandas + DuckDB** (+ polars optional) inside executed code; suits KramaBench pipelines.

### 3.12 Schemas & Config

**Pydantic v2** for all structured objects (TransactionUnit, Decision, Evidence, ValidationReport, MemoryOp) + YAML/TOML thresholds config.

### 3.13 Evaluation Benchmarks

| Benchmark         | Role                                                                |
| ----------------- | ------------------------------------------------------------------- |
| **KramaBench** ✅ | Primary (paper's main results): 104 NL tasks, 24 sources, 6 domains |
| DA-Code           | Secondary comparison (DA-Agent lineage)                             |
| AgenticDataBench  | Paper's consistency experiments; optional                           |

Metrics: score (%), #code steps, tokens, $cost, consistency = √(avg per-task variance over 3 runs).

---

## 4. RECOMMENDED DEFAULT STACK (v2 one-liner)

> **Python 3.11 · LangChain (models/prompts/tools) + LangGraph (transactional state machine, PostgresSaver, Send API) · OpenCode headless as the coding/execution backbone · swappable backbone LLM via OpenAI-compatible APIs · Qwen3-0.6B on vLLM for confidence scoring · PostgreSQL (+pgvector, optional AGE) for checkpoints/events/memory/skills · GitPython append-only workspace · Docker for competitive isolation · typer+pytest skill hub · LangSmith tracing · KramaBench eval.**

---

## 5. KEY MECHANISMS & PARAMETERS (from the paper)

- **Confidence(C)** = exp(mean token-level log-prob of output C), scored by the local model.
- **Divergence(a, b)** = compare confidences of the same target under two contexts (with vs. without evidence).
- **Decision divergence**: explored decisions (from evidence summary) vs executed decisions (extracted from code); low divergence ⇒ executed choice isn't better-evidenced ⇒ retry.
- **Code divergence**: AST-extract decision-relevant spans (control flow); score with vs. without evidence; max span divergence < 0.50 ⇒ retry.
- **Exploration redundancy**: summarize new observation ± prior observations; divergence > 0.45 ⇒ stop exploring.
- Budgets: **≤20 units/task · retain 15 historical units · ≤2 retries/unit · exploration 1–4 rounds (-1 every 2 units)**.

---

## 6. BUILD PHASES

1. [ ] **Scaffolding** — repo layout, Pydantic schemas, config; `docker-compose` for Postgres (+pgvector); LangGraph skeleton (task-level + unit-level graphs); LangSmith hookup.
2. [ ] **OpenCode runner** — headless invocation wrapper (scoped prompts, stdout/diff parsing), git snapshot → run → validate → commit-or-revert staging discipline.
3. [ ] **Transactional state** — Postgres schema (events/WAL, runs, units, validations); wire `AsyncPostgresSaver`; rollback = git revert + event tombstones.
4. [ ] **Confidence engine** — vLLM-served Qwen3-0.6B scoring service; three divergence measures; threshold config.
5. [ ] **Core loop** — full explore → decisions → codegen → validation gate → retry judge → commit in LangGraph; plain ReAct baseline (LangGraph `create_react_agent`) as ablation control.
6. [ ] **Isolation layer** — `Send`-based independent agents; branch-per-agent collaborative flow; Docker-isolated competitive runners.
7. [ ] **Durability layer** — memory_nodes/edges tables + LLM evolution ops; provenance-enriched history; replay-from-checkpoint recovery demo.
8. [ ] **Skill hub** — package repos as CLI skills, generate pytest suites, pgvector-powered router.
9. [ ] **Evaluation** — KramaBench runner, 3 runs/task, Table-2/3/4-style reporting vs baseline.

**Deferred (paper's "envisioned" parts):** fine-tuned isolation-policy tuner; consistency-oriented benchmarks; automatic memory-training supervision from trajectories.

---

## 7. RISKS & MITIGATIONS

| Risk                                       | Mitigation                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| LangGraph API churn between versions       | Pin versions; isolate graph definitions in one module                          |
| OpenCode headless output parsing fragility | Prefer its structured/SDK output modes; fall back to git-diff-based extraction |
| Provider lacks logprobs                    | Local 0.6B confidence model (already planned)                                  |
| Retry loops explode cost                   | Hard caps (2 retries/unit, 20 units), token budget guard                       |
| Postgres becomes required infra            | Ship `docker-compose.yml`; graceful SQLite fallback for dev                    |
| OpenCode subprocess hangs                  | Timeouts + kill-and-revert-to-snapshot semantics                               |
| Git operations slow under many commits     | Batch artifact commits; pygit2 if needed                                       |
| Confidence miscalibration (small model)    | Calibrate on held-out tasks; self-consistency fallback for critical gates      |
| Docker unavailable                         | Subprocess + temp dirs fallback for competitive mode                           |

---

## 8. OPEN DECISIONS (confirm before Phase 1)

1. **Harness confirmation**: OpenCode OK as the coding backbone? And what did you mean by **"freebuff"** — I can't find a harness by that name; did you mean **Goose**, **Aider**, **Cline**, or something else?
2. **Backbone LLM provider + API key** (Bailian/Qwen? OpenRouter? local vLLM?) — this configures both OpenCode and LangChain clients.
3. **Postgres**: OK to run via docker-compose locally? Any existing instance/connection string?
4. **GPU** available for Qwen3-0.6B confidence server, or CPU-only acceptable?
5. **Benchmark scope**: full KramaBench (104 tasks) or a subset (e.g., Environment domain, as in the paper's tables)?
6. **Tracing UI**: LangSmith (cloud, needs key) vs Langfuse (self-hosted) vs Postgres-events-only?

---

## 9. IMPLEMENTATION DECISIONS LOCKED (v2.1)

Resolved before build kickoff:

1. **Harness**: OpenCode (already installed at `~/.opencode/bin/opencode`) driven headlessly. "freebuff" never identified — proceeding with OpenCode.
2. **Backbone LLM**: user has an **OpenAI or Anthropic direct key** → config supports both via `LLM_PROVIDER=openai|anthropic`; key goes in `.env`.
3. **PostgreSQL**: via docker-compose (Docker 29.7.2 confirmed).
4. **GPU (confirmed via nvidia-smi)**: NVIDIA MX450, 2 GB VRAM (~1.8 GB free).
   - **vLLM dropped** — impractical at 2 GB.
   - **Confidence engine = Qwen3-0.6B via HuggingFace `transformers`** (FP16, ~1.2 GB weights) on the MX450 with automatic CPU fallback. Pure-pip install, exact token logprobs, short scoring texts fit VRAM.
   - Avoid aggressive quantization for scoring fidelity; recalibrate thresholds after first runs.
5. **Benchmark scope**: start with a small built-in task suite (KramaBench-style); real KramaBench hookup documented as an extension.
6. **Tracing**: Postgres `events` table + JSONL files; LangSmith optional via env flag.
