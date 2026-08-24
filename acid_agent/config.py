"""Central configuration. All values come from .env (see .env.example)."""

from functools import lru_cache

import psycopg
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Backbone LLM
    llm_provider: str = "openai"  # openai | anthropic
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    openai_base_url: str = ""  # e.g. https://api.deepseek.com for DeepSeek
    anthropic_api_key: str = ""

    # OpenCode
    opencode_bin: str = "opencode"
    opencode_timeout_s: int = 300

    # Postgres
    database_url: str = "postgresql://acid:acid@localhost:5433/acid_agent"

    # Local confidence model
    confidence_model: str = "Qwen/Qwen3-0.6B"
    confidence_device: str = "auto"

    # Budgets & thresholds (paper defaults)
    max_units: int = 20
    memory_window_units: int = 15
    max_retries_per_unit: int = 2
    exploration_max_rounds: int = 4
    exploration_min_rounds: int = 1
    redundancy_threshold: float = 0.45
    decision_divergence_min: float = 0.25
    code_span_divergence_min: float = 0.50


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_conn() -> psycopg.Connection:
    """Plain sync connection for tracer/memory/skills."""
    return psycopg.connect(get_settings().database_url)