"""Backbone LLM access via LangChain (OpenAI or Anthropic, chosen by LLM_PROVIDER)."""

import json
import re

from langchain_core.language_models import BaseChatModel

from .config import get_settings


def get_backbone() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=s.llm_model,
            api_key=s.anthropic_api_key,
            temperature=0,
            timeout=120,
            max_retries=2,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.llm_model,
        api_key=s.openai_api_key,
        base_url=s.openai_base_url or None,
        temperature=0,
        timeout=120,
        max_retries=2,
    )


def ask(prompt: str) -> str:
    """Plain text completion from the backbone."""
    return get_backbone().invoke(prompt).content


def ask_structured(prompt: str, schema):
    """Structured output parsed into a pydantic schema.

    Tries provider-native methods first (function_calling works on DeepSeek,
    OpenAI, Anthropic). Falls back to manual JSON parsing for providers that
    reject response_format json_schema.
    """
    model = get_backbone()
    for method in ("function_calling", "json_schema"):
        try:
            return model.with_structured_output(schema, method=method).invoke(prompt)
        except Exception as e:
            last_err = e
    # Manual fallback: ask for JSON, parse locally
    schema_json = (
        json.dumps(schema.model_json_schema(), indent=2)
        if hasattr(schema, "model_json_schema")
        else "{}"
    )
    raw = ask(
        prompt + "\n\nRespond with ONLY one JSON object matching this JSON Schema "
        "(no markdown, no commentary):\n" + schema_json
    )
    try:
        return schema.model_validate_json(_extract_json(raw))
    except Exception as e:
        raise RuntimeError(f"structured output failed: {last_err} | fallback parse failed: {e}") from e


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text.strip()