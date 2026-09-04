# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Implementation of the paper *"Agentic Transactions: Towards ACID-Compliant Agent Systems"* (Sun, Wang, Li). A data-analysis agent where each unit of work is a **semantic transaction**: explore → extract decisions → generate+execute → validate → commit-or-rollback. The research claim being tested is that the transactional gate improves accuracy *and* run-to-run consistency versus the same model with no gate, so the repo is built as an A/B: `claude-acid` (full machinery) vs `claude` (one raw headless session, same model, same workspace).

`Plan.md` is the original design blueprint and `README.md` the user-facing intro; both drift. Where any of them disagrees with the code, trust the code. (Known drift: README says real KramaBench is "not bundled" — `eval/kramabench_tasks.py`, `vendor/`, and `data/kramabench/` now provide it.)

## Commands

```bash
source .venv/bin/activate
pip install -e ".[dev]"           # core + pytest
pip install -e ".[confidence]"    # torch/transformers for the local Qwen3-0.6B-Base scorer

docker compose up -d              # Postgres+pgvector on localhost:5433; db/schema.sql auto-loads on FIRST start only

pytest -q                          # tests/ only (testpaths is pinned); graph-wiring tests skip without Postgres
pytest tests/test_workspace.py -q  # single file
pytest tests/test_confidence_math.py::test_surprise_is_one_sided -q  # single test
pytest skills -q                   # skill suites are OUTSIDE testpaths — must be named explicitly

python scripts/smoke_llm.py            # verify the claude-CLI backbone + JSON structured output work
python scripts/setup_confidence.py     # download Qwen3-0.6B-Base and self-test signal DIRECTION

python scripts/acid_cli.py run-task "<question>" --agent claude-acid --data-dir path/to/csvs
python scripts/acid_cli.py run-task "<question>" --agent claude        # harness baseline
python scripts/acid_cli.py run-task "<question>" --agent claude-react  # bare ReAct baseline
python scripts/acid_cli.py kramabench domains                          # which domains have data on disk
python scripts/acid_cli.py eval --agent claude-acid --domain archeology --runs 3 --task-ids 0-5
python scripts/acid_cli.py eval --agent claude-acid --synthetic --runs 3   # built-in deterministic tasks

python scripts/compare_arms.py                       # A/B table from results/ (safe to run mid-eval)
python scripts/regrade.py --suite krama --domain archeology --write   # re-score history, no re-runs
```

`db/schema.sql` only runs at container init. Schema changes need `docker compose down -v` (destroys data) or manual `psql` — the file already carries one in-place `ALTER TABLE` migration guard as precedent.

## Architecture

Two graphs, one gate, three persistence layers.

**`runner.run_task`** is the single entry point: creates run_id + `Workspace` + `Tracer`, writes the `runs` row, dispatches on `agent_type`, and marks the run done/failed. All arms share the workspace and the model. The `runs` row must exist before any unit work — `units.run_id` is a FK to it.

**Three arms, and the difference between them is the point of the repo:**

| `agent_type` | what runs the loop | maps to the reference's |
|---|---|---|
| `claude-acid` (`acid`) | task_graph → unit_graph, full transaction machinery | `acid` |
| `claude` | one Claude Code session (`--dangerously-skip-permissions`, `cwd=ws.root`) | `claude-code` |
| `claude-react` (`react`) | `baseline_react.py` owns the Thought/Action/Observation loop; the model is reached only through stateless `llm.ask` | `prompt` |

`claude` and `claude-react` are **both** no-machinery controls, but they are not interchangeable. `claude` includes the Claude Code harness — its own planning, file access and self-correction — which is capability the reference's `prompt` arm never had, so it sits high (89.9% on the easy tier) and leaves the gate little to win. `claude-react` strips the harness: `llm.ask` is a plain `claude -p` with no `cwd` and no tool access, so the model can only touch the workspace through actions this repo parses and executes. Report which control a number is against; "ACID vs Claude Code" and "ACID vs a bare ReAct loop" are very different claims.

Every arm authenticates the same way — the CLI's subscription login. No arm needs an API key. (The reference's `claude-code` arm defaults `ANTHROPIC_BASE_URL` to a DashScope endpoint, i.e. the Claude Code harness driven by a Qwen model; that default is theirs alone and `claude_env()` strips any inherited base-URL/token so it cannot leak in here.)

**`graphs/task_graph.py` (supervisor)** — `plan_units` → `run_unit`* → `assemble_answer`. Owns the budgets: ≤ `max_units` units (planning asks for ≤5), exploration budget decays one round every two units (`exploration_max_rounds - i//2`), and an early-completion check after each unit. Only *committed* unit outputs flow into `evidence_summary` for later units; failed units contribute nothing but a `(unit failed; excluded)` placeholder in `results`.

**`graphs/unit_graph.py` (the transaction)** — `explore` → `extract_decisions` → `generate_execute` → `validate` → {`commit` | `retry_rollback` → back to `extract_decisions` | `fail_unit`}. Four details matter:
- Retries re-enter `extract_decisions`, not `generate_execute`, so gate feedback can revise the *decisions*, not just the code.
- Both retry and failure call `ws.rollback()` (`git reset --hard` + `clean -fd`). That is the atomicity guarantee: a rejected attempt leaves zero trace in the workspace.
- `explore` has a **fast path**: the backbone writes one read-only profiling snippet that `ws.run_code` executes directly (~5–12s). A full `run_claude` session is the fallback, used only when the snippet fails *and* produced no stdout. Both feed the same summarizer, so the gate can't tell which ran.
- `generate_execute` runs the script **twice on purpose**: the claude session writes and runs `unit{i}.py` inside the workspace, then `ws.run_script` re-runs it for a clean, deterministic exec signal for the gate. The filename `unit{i}.py` is a hard contract — if the session writes anything else, `code` reads back empty and the backbone-codegen fallback replaces it.

**`validation.py` (the gate)** — four components, each scored `pass` / `watch` / `retry` / `skipped`, merged so that **only a `retry` fails the attempt**. `watch` is recorded on `report.watchlist` and lets the attempt through, so a soft signal cannot burn the retry budget alone.

| component | source | red when |
|---|---|---|
| `execution_observation` | exit code / stdout / stderr | non-zero exit, or exit 0 with **no stdout** (`watch` if stdout + stderr) |
| `reasoning_observation` | backbone LLM reflection | reviewer says not ok |
| `probability_contrast` | small LM, **only on LLM-flagged conflicts** | `P(current)/P(alternative) < 0.25` (`watch` < 0.75) |
| `evidence_surprise` | small LM over AST code spans | **depends on `gate_semantics`** — see below |

**The paper and the authors' released code disagree on the code-span rule**, and `Settings.gate_semantics` selects which one decides:

| | metric | retry when | source |
|---|---|---|---|
| `paper` (default) | `\|C_with − C_without\| / max(...)` | **below** `span_divergence_min` (0.50) — evidence changed nothing, so the code is not grounded in it | paper §2.2.2 |
| `reference` | `max(0, −(logp_with − logp_without))` nats | **above** `span_surprise_retry` (0.50), watch above 0.10 — evidence suppresses the span, so the code contradicts it | `evidence_surprise.py:544-552` |

Same 0.50 threshold, opposite direction. **Both metrics are computed and persisted on every attempt** (`max_span_divergence`, `max_span_surprise`, plus `gate_semantics` recording which one ruled), so one run yields the counterfactual for the other rule — `scripts/gate_stats.py` prints it. The decision signal has no such split: the paper's "executed vs explored-alternative decision under the same context, retry below 0.25" is exactly `probability_contrast.py`, so both agree there.

**Calibration warning (measured 2026-09-04, Qwen3-0.6B-Base).** The paper's 0.50 is not portable to this scorer. On a script that correctly implements every evidence item, `max_span_divergence` came out **0.303** conditioned on the code prefix and **0.426** standalone — both below 0.50, so the paper rule rejects well-grounded code and would exhaust `max_retries_per_unit` on every unit. Calibrate `span_divergence_min` against a real distribution (run a few units with `GATE_BYPASS=1`, which records all signals while forcing PASS) before trusting a paper-mode result.

`decision_surprise` is computed and persisted but **never gates** — the reference calls its equivalent "diagnostic-only". Every signal fails **open**, which is worth remembering when a run looks suspiciously permissive:
- confidence model missing/unloadable ⇒ those components are `skipped`, not red ⇒ the gate runs on execution + reflection only;
- no LLM-flagged evidence/code conflict ⇒ `probability_contrast` is `skipped` (this is the common case, by design);
- reflection call raises ⇒ `reflection_ok = True`;
- `code_span_surprise` reports `available=False` when the AST finds no decision-relevant span, which reads as `skipped` (trivial or unparseable scripts auto-pass that component).
Each attempt is persisted to `validations` (best-effort; a DB failure is swallowed). Reflection is scoped to the *unit goal*, not the final answer — the prompt explicitly tells the reviewer not to demand the whole task.

**`confidence.py`** — everything is a **mean token log-prob in nats**, compared as a **raw delta**. Nothing is exponentiated before thresholding. `score_logp` is the primitive; `score` (the exp form) exists for reporting only — never threshold on it, because exp() of a small model's mean logprob lives near zero, which is exactly what forced the old relative normalization.

Four probes, mirroring `vendor/acid-paper-ref/da_agent/utils/`:
- `code_span_surprise` (their `evidence_surprise.py`) — `max(0, -(logp(span | task+evidence+code_prefix) - logp(span | task+code_prefix)))`. **High is bad**: evidence suppressing a span means the code contradicts what exploration found. One-sided on purpose — evidence *supporting* a span clamps to 0. Each span is conditioned on the code that precedes it.
- `probability_contrast` (theirs, same name) — scores the policy the code implements against the evidence-backed alternative as two continuations of the *same* prefix; `ratio = exp(delta)`. Probes exist **only** for conflicts `validation._alignment_conflicts` already extracted from the backbone, so the small model referees a disagreement the strong model found rather than scanning everything.
- `decision_surprise` (their `anchor_decision_surprise.py`) — same delta form over decisions. **Diagnostic only.**
- `exploration_redundancy` (their `adaptive_exploration_metric.py`) — PMI/token, `logp(obs | priors) - logp(obs)`. High ⇒ already predicted ⇒ stop exploring.

**Measured on Qwen3-0.6B-Base (2026-09-04), `scripts/setup_confidence.py`:** adding
the exploration summary to the context raises the likelihood of *most* spans
(deltas of +0.13 to +0.16 nats on a date-parsing example, for both a span that
follows the evidence and one that contradicts it). Because `span_surprise` is
one-sided — `max(0, -delta)`, exactly as the reference defines it — both clamp to
0.0 and the component reads `pass`. The *ordering* still carries signal (the
aligned span gained +0.1549 vs the contradicting span's +0.1253), but the clamp
discards it. Expect `evidence_surprise` to be quiet and the gate to lean on
execution + reflection + probability_contrast. Do not "fix" this by dropping the
clamp — that is the reference's definition; treat a quiet component as a finding
about the 0.6B scorer, not a bug.

The model is **Qwen3-0.6B-Base**, not the instruct checkpoint: every signal is a raw likelihood comparison and instruct tuning reshapes the token distribution. Scoring is tuned for a ~1.6 GB GPU: log-softmax is sliced to target rows and chunked at 64, and a mid-scoring CUDA OOM migrates the model to CPU and re-scores rather than propagating — an escaping OOM would be swallowed by the gate and silently turn validation off mid-run.

**Tracing**: two layers. `tracer.py` is the paper's own append-only provenance log (JSONL + Postgres `events`) and always runs. `tracing.py` is optional LangSmith observability — `configure()` runs at package import and pushes `LANGSMITH_*` from Settings into `os.environ`, which is the only place the SDK looks; `.env` alone does nothing. It writes `LANGSMITH_TRACING` either way, so tracing=true without a key is forced off instead of retrying against 401s. LangGraph nodes auto-trace; the `claude -p` subprocess calls are invisible to LangChain callbacks so they carry explicit `@traced` decorators (`backbone.ask`, `backbone.ask_structured`, `claude_code.session`, `validation_gate`, `acid_task`). `drop=`/`summarize=` keep seed-file bytes and pydantic classes out of trace payloads. Don't confuse the two modules.

**Persistence**: git workspace (`workspaces/<slug>/`, atomicity) · Postgres (`runs`/`units`/`validations`/`events`/`memory_nodes`/`memory_edges`/`skills`, durability) · JSONL trace (`runs/<run_id>/events.jsonl`). `Tracer.log` writes JSONL first and swallows DB errors — tracing must never break a run.

**`memory.py`** — after each *committed* unit, an LLM proposes insert/merge/delete ops over the knowledge graph. Scoped by `task_slug` so eval tasks don't cross-contaminate. Failed units never touch it, and `memory.evolve_from_unit` raising is caught inside `commit` so a memory failure can't undo a validated commit.

## The two LLM call paths

Everything routes through the `claude` CLI on subscription auth — `claude_env()` strips any inherited `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` so a stale export can't redirect it, and sets `ANTHROPIC_MODEL` only when `CLAUDE_MODEL` is non-empty. The two call paths are not interchangeable:

- **`llm.ask` / `llm.ask_structured`** — backbone reasoning (planning, decision extraction, reflection, memory evolution, exploration snippets). Plain `claude -p`, no `cwd`, no permission bypass, `MAX_THINKING_TOKENS=2048`, 360s timeout.

  **`claude -p` is not a tool-free completion.** Verified 2026-09-04: it retains Read/Glob/Grep and will go find files on its own. Because `_run_cli` passes no `cwd`, these calls run in the **repo root** — so a backbone call can reach `workspaces/`, `data/`, and the rest of the repo. Consequences worth keeping in mind: the reflection judge can read the workspace it is judging, and the `explore` snippet writer can inspect the data before writing its "blind" profiling script. Neither is currently prevented.

  Flags do **not** fix this. `--disallowedTools "Bash,Read,Glob,..."` was observed satisfying the request through a different tool anyway; `--allowedTools ""`, `--allowedTools NoSuchTool`, `--permission-mode manual --permission-prompts none` and `--restricted` alone all still read the file.

- **`llm.ask_isolated`** — the version that actually holds: `--restricted --strict-mcp-config` **with `cwd` set to a fresh empty temp dir**. Restricted mode drops the code-running tools and confines the file tools to the working directory; an empty working directory then leaves nothing to read. Confirmed to refuse an absolute path outside its cwd. Used by `baseline_react.py`, where a model that finds the data itself would invalidate the arm entirely.

  There is **no native structured output** on any of these: `ask_structured` injects the pydantic JSON Schema into the prompt, extracts the object (fenced block, else first `{`…last `}`), validates, and retries once. Assume it can raise.
- **`claude_runner.run_claude`** — an agentic session with file access: `--dangerously-skip-permissions`, run with `cwd=ws.root`, `claude_timeout_s` (900s). Used for `generate_execute`, the `explore` fallback, and the baseline arm. It never raises — timeouts and a missing binary come back as `ExecResult(ok=False)`.

Because `llm._run_cli` passes no `cwd`, backbone calls run in the repo root — the stray `unit0.py`/`unit1.py`/`unit2.py` at the top level are leaked artifacts from that, not source.

Nodes that call these are wrapped in try/except and degrade rather than crash: exploration rounds break out, codegen falls back to `ask`, completion checks default to "not complete", redundancy checks pass silently.

## Config

`config.py:Settings` (pydantic-settings) is the only config surface; `.env` keys map to field names case-insensitively. There are no API keys for the backbone — it authenticates as the logged-in Claude Code CLI.

**Read `.env` before reasoning about behavior; it is the source of truth and differs from the `config.py` defaults.** Current on-disk divergences: `CONFIDENCE_DEVICE=cuda` (vs `auto`), `CLAUDE_MODEL` pinned to a specific model (vs empty, i.e. CLI default), and LangSmith tracing **on** with a live key — so ordinary runs are being exported to LangSmith. `get_conn()` opens a plain sync psycopg connection; there is no pool.

## Evaluation

Two suites, two result shapes.

- `eval/kramabench.py` — three self-generating synthetic tasks with graders (including a mixed-date-format trap). Reached via `eval --synthetic`; writes `results/eval_<agent>_<ts>.json` with a `per_task_mean` key.
- `eval/kramabench_tasks.py` — the *real* benchmark: workloads from `vendor/acid-paper-ref/Kramabench/workload/<domain>.json`, inputs from `data/kramabench/data/<domain>/input/`. Default path for `eval` and `kramabench run`; writes `results/krama_<agent>_<domain>_<ts>.json` with a `per_task` key. `vendor/` is gitignored — missing data means `get_available_domains()` returns fewer domains, not an error. All 6 domains currently have data.

**Grading is the main source of fake results here — verify a 0.0 before believing it.** Both suites now agree on two rules, and both were bugs that produced false zeros on *correct* answers:
- *Numeric*: scan **every** number in the answer, pass if any is within tolerance (synthetic 2%, KramaBench 0.005). Taking the first number scored "across all 99 rows the average is 5.28" against `99`.
- *String/list*: compare casefolded and diacritic-stripped (`_norm`), because agents write "Sao Paulo" for "São Paulo".

Both eval paths **checkpoint after every run** (`runs_completed` says how far a partial file got), so an interrupted multi-hour eval keeps what it earned. Because `runs.final_answer` is in Postgres, grading is a pure function applied after the fact: `scripts/regrade.py` re-scores completed runs with the current graders instead of re-running the agent. Reach for it whenever a grader changes.

Seed files are read as **bytes** for KramaBench (workloads mix CSV with `.xlsx`; a text round-trip corrupts them) but as **text** in the `run-task` CLI path, which therefore cannot seed binary inputs. `Workspace.create` handles both.

`KramaTask.seed_files` resolves each `data_sources` entry in four ways — exact relative path, glob expansion (`"State MSA Identity Theft Data/*"`), directory recursion, then a shallowest-match basename search. All four are needed: `legal/` nests its 135 files in subdirectories, so exact-path matching found **zero** of them and every legal task silently seeded an empty workspace and scored 0.00. `get_available_domains()` only checks that the workload JSON and input directory exist, so it still reports such a domain as available — a domain listing as available does not mean its tasks can load data. After the fix all 42 easy tasks across the 6 domains seed successfully.

The reported metrics are `overall_score` and the paper's consistency metric `sqrt(mean per-task variance across runs)` — which is why `--runs` defaults to 3 and slugs are `<prefix>_<agent>_<task>_r<n>`. The `_r<n>` matters: without it every repetition shares one workspace *and* one memory scope, so the consistency metric measures contamination. Consistency is 0.0 for a single run by construction, and with 3 binary runs per-task variance can only be 0, 0.222 or 0.333 — treat it as directional.

**Measured costs and known ceilings/floors** (2026-09-03, `CLAUDE_MODEL=claude-sonnet-4-6`):
- ACID arm ≈ 15–20 min per task-run (19.5 min on `archeology-hard-1`: 4 units, 6 attempts); baseline ≈ 2 min. Budget the A/B at roughly 8:1.
- The synthetic suite is at **ceiling**: baseline scores 100% (9/9), including the `mixed_dates` trap — a current model reasons that dates are irrelevant to a temperature average and never does the naive parse. The gate cannot show a gain where baseline is already perfect, so don't run the A/B there.
- `archeology-hard-1` is at **floor**: both arms scored 0.0. Pick tasks where baseline is *imperfect but not hopeless*; run the cheap baseline arm first to find that band before spending ACID time.

## Testing notes

`tests/test_graph_wiring.py` fakes the whole outside world by monkeypatching **module-level names**: `unit_graph.ask`, `unit_graph.run_claude`, `unit_graph.ask_structured`, `validation.ask_structured`, `confidence.code_span_surprise`, `confidence.probability_contrast`, `confidence.decision_surprise`. The confidence fakes must return the probe **dicts** (`{'available': ..., 'max_surprise': ...}`), not bare floats. A new LLM/exec call site must be imported into the module namespace (`from ..llm import ask`) rather than called through a package path, or it will escape the fakes and make a live CLI call during tests. The fakes also dispatch on `schema.__name__` and on prompt substrings (`"Condense"`), so renaming `Decisions` or reworking the summarizer prompt breaks them.

## Wired vs. present-but-unused

`skills.py` (skill hub + LLM router + pytest-gated skill validation) and `isolation.py` (independent / collaborative-branch / competitive-clone sub-agent spawning) are implemented and reachable from the CLI/tests, but **neither is called by `runner.run_task`**. The live path is task_graph → unit_graph only. Don't assume a skill lookup or sub-agent spawn happens during a normal run. Likewise the `skills.embedding` pgvector column and the `exploration_redundancy` field on `ValidationReport`/`validations` exist but nothing ever populates them.

`workspaces/`, `runs/`, `results/`, `.env`, and `vendor/` are gitignored; the many `workspaces/krama_*` directories are prior eval output, not code.
