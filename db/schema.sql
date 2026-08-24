-- ACID-Agent schema. Loaded automatically on first container start.
CREATE EXTENSION IF NOT EXISTS vector;

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
    decision_divergence      REAL,
    max_code_span_divergence REAL,
    exploration_redundancy   REAL,
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

-- Skill hub registry (pgvector for routing; dimension matches OpenAI text-embedding-3-small)
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