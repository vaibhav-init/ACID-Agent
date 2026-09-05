"""Central configuration. All values come from .env (see .env.example)."""

from functools import lru_cache

import psycopg
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Headless agent session (execution backbone for BOTH arms).
    # Auth is the opencode go subscription login — no API key or router lives here.
    opencode_bin: str = "opencode"
    opencode_model: str = "opencode-go/qwen3.7-plus"
    session_timeout_s: int = 900

    # Postgres
    database_url: str = "postgresql://acid:acid@localhost:5433/acid_agent"

    # Local confidence model. The -Base checkpoint is what the reference uses:
    # instruct tuning reshapes the token distribution, and every signal here is a
    # raw likelihood comparison, so the base model is the calibrated choice.
    confidence_model: str = "Qwen/Qwen3-0.6B-Base"
    confidence_device: str = "auto"

    # LangSmith tracing (optional; exported to os.environ by tracing.configure())
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "acid-agent"

    # Budgets & thresholds (paper defaults; span threshold calibrated, see .env)
    max_units: int = 20
    memory_window_units: int = 15
    max_retries_per_unit: int = 2
    exploration_max_rounds: int = 4
    exploration_min_rounds: int = 1
    redundancy_threshold: float = 0.45   # PMI/token in nats; high => redundant

    # --- Code-span gate: the paper and the authors' released code disagree ---
    # Paper §2.2.2: "the maximum code-span confidence divergence is below 0.50"
    #   => divergence is |C_with - C_without| and a LOW value means the code is
    #      unchanged by the evidence, i.e. ungrounded => retry.
    # Reference evidence_surprise.py: surprise = max(0, -(logp_with - logp_without))
    #   => a HIGH value means the evidence SUPPRESSES the span, i.e. the code
    #      contradicts what exploration found => retry.
    # Same 0.50 threshold, opposite direction. Both metrics are always computed
    # and persisted; this selects which one decides the verdict.
    gate_semantics: str = "paper"  # "paper" | "reference"

    # paper mode (relative divergence in [0,1]; retry BELOW)
    span_divergence_min: float = 0.50

    # reference mode (raw nats; retry ABOVE, watch band below it)
    span_surprise_warn: float = 0.10
    span_surprise_retry: float = 0.50

    # Decision signal — paper and code AGREE here, so there is nothing to switch.
    # Paper: executed vs explored-alternative decision under the SAME context,
    # retry below 0.25. Reference probability_contrast.py: identical.
    contrast_warning_ratio: float = 0.75  # P(current)/P(evidence-backed alt)
    contrast_retry_ratio: float = 0.25
    max_contrast_probes: int = 4
    max_code_spans: int = 10

    # Diagnostic only, never gates (reference: anchor_decision_surprise.py).
    decision_surprise_warn: float = 0.10
    decision_surprise_retry: float = 0.50

    # ReAct baseline (agent_type "react"): the paper's `prompt` arm shape.
    # Defaults match the reference's run_kramabench.py (--max_steps 20,
    # --max_memory_length 15).
    react_max_steps: int = 20
    react_max_memory: int = 15
    react_action_timeout_s: int = 180

    # opencode go subscription; enables the Qwen model family — the paper's
    # backbone — for regime experiments at zero extra cost
    # opencode_model lives with the session settings above.

    # Ablation: force every gate verdict to PASS while still computing and
    # persisting all four signals. Isolates the transaction scaffolding
    # (decompose + explore) from the validation gate itself. Never set in .env.
    gate_bypass: bool = False

    # Interpretation review (2026-09-04, targets the hard-tier failure mode):
    # proactive — before first codegen, an LLM probe challenges the FIRST reading
    # of an ambiguous unit goal (distance metric, units, inclusion criteria) and
    # may revise the decisions. Reactive — per-attempt answer candidates are
    # compared when two attempts disagree (port of the reference's
    # AnswerCandidateTracker, which the repo had never wired).
    interpretation_review: bool = True
    answer_candidates: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_conn() -> psycopg.Connection:
    """Plain sync connection for tracer/memory/skills."""
    return psycopg.connect(get_settings().database_url)
