-- ACID-Agent schema. Loaded automatically on first container start.
CREATE EXTENSION IF NOT EXISTS vector;

-- Migration: add task_slug to memory_nodes for per-task memory scoping
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='memory_nodes' AND column_name='task_slug') THEN
        ALTER TABLE memory_nodes ADD COLUMN task_slug TEXT;
    END IF;
END
$$;

-- Migration: gate signals moved from relative divergences to raw log-prob
-- surprise/ratio (see acid_agent/confidence.py). Old columns are kept so
-- historical rows stay readable; new runs write the new ones.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='max_span_surprise') THEN
        ALTER TABLE validations ADD COLUMN max_span_surprise REAL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='decision_surprise') THEN
        ALTER TABLE validations ADD COLUMN decision_surprise REAL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='contrast_min_ratio') THEN
        ALTER TABLE validations ADD COLUMN contrast_min_ratio REAL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='review_decision') THEN
        ALTER TABLE validations ADD COLUMN review_decision TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='watchlist') THEN
        ALTER TABLE validations ADD COLUMN watchlist JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='max_span_divergence') THEN
        ALTER TABLE validations ADD COLUMN max_span_divergence REAL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='validations' AND column_name='gate_semantics') THEN
        ALTER TABLE validations ADD COLUMN gate_semantics TEXT;
    END IF;
END
$$;

-- Append-only event log / WAL + provenance
CREATE TABLE IF NOT EXISTS events (
    id       BIGSERIAL PRIMARY KEY,
    run_id   UUID NOT NULL,
    ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
    type     TEXT NOT NULL,          -- e.g. llm_call | tool_call | explore | validate | commit | rollback
    payload  JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_run_idx ON events (run_id, id);

-- One row per agent run over a task
CREATE TABLE IF NOT EXISTS runs (
    run_id      UUID PRIMARY KEY,
    task_text   TEXT NOT NULL,
    agent_type  TEXT NOT NULL DEFAULT 'acid',   -- acid | baseline
    status      TEXT NOT NULL DEFAULT 'running',-- running | done | failed
    final_answer TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

-- One row per semantic transaction unit
CREATE TABLE IF NOT EXISTS units (
    id            BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES runs(run_id),
    unit_index    INT  NOT NULL,
    goal          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running', -- running | committed | failed
    attempts      INT  NOT NULL DEFAULT 0,
    summary       TEXT,
    git_commit    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, unit_index)
);

-- Validation gate results (one row per attempt)
CREATE TABLE IF NOT EXISTS validations (
    id                       BIGSERIAL PRIMARY KEY,
    run_id                   UUID NOT NULL,
    unit_index               INT  NOT NULL,
    attempt                  INT  NOT NULL,
    decision_divergence      REAL,   -- legacy (relative divergence), pre-log-space gate
    max_code_span_divergence REAL,   -- legacy (relative divergence), pre-log-space gate
    exploration_redundancy   REAL,
    max_span_surprise        REAL,   -- reference rule; nats; high => code contradicts evidence
    max_span_divergence      REAL,   -- paper rule; [0,1]; low => code not grounded in evidence
    gate_semantics           TEXT,   -- which rule decided this verdict: paper | reference
    decision_surprise        REAL,   -- nats; diagnostic only, never gates
    contrast_min_ratio       REAL,   -- P(current)/P(evidence-backed alternative)
    review_decision          TEXT,   -- pass | watch | retry
    watchlist                JSONB,
    execution_ok             BOOLEAN,
    reflection_ok            BOOLEAN,
    passed                   BOOLEAN NOT NULL,
    feedback                 TEXT,
    ts                       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Transaction-aware memory graph (durability)
CREATE TABLE IF NOT EXISTS memory_nodes (
    id         BIGSERIAL PRIMARY KEY,
    key        TEXT NOT NULL,
    content    TEXT NOT NULL,
    props      JSONB NOT NULL DEFAULT '{}',
    source_run UUID,
    task_slug  TEXT,  -- scope memory to a specific task; NULL = global
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (key)
);
CREATE TABLE IF NOT EXISTS memory_edges (
    id      BIGSERIAL PRIMARY KEY,
    src_key TEXT NOT NULL REFERENCES memory_nodes(key) ON DELETE CASCADE,
    dst_key TEXT NOT NULL REFERENCES memory_nodes(key) ON DELETE CASCADE,
    rel     TEXT NOT NULL,
    props   JSONB NOT NULL DEFAULT '{}',
    UNIQUE (src_key, dst_key, rel)
);

-- Skill hub registry (pgvector column reserved for routing; unused until an embedding source is wired)
CREATE TABLE IF NOT EXISTS skills (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL,
    path          TEXT NOT NULL,
    embedding     VECTOR(1536),
    use_count     INT NOT NULL DEFAULT 0,
    success_count INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);