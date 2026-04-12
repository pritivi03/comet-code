"""Hardcoded catalog of OpenRouter models exposed to the user.

This is intentionally a small curated list rather than a full pull from
the OpenRouter API — keeps the UI snappy and gives us known-good defaults.
Identifiers follow OpenRouter's `<vendor>/<model>` slug format.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    slug: str           # OpenRouter slug, e.g. "anthropic/claude-sonnet-4.5"
    short: str          # Short alias the user can type, e.g. "sonnet"
    label: str          # Human-friendly display name


AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo("google/gemma-4-31b-it:free", "gemma", "Gemma 4 31B"),
    ModelInfo("anthropic/claude-sonnet-4.5", "sonnet",  "Claude Sonnet 4.5"),
    ModelInfo("anthropic/claude-opus-4.1",   "opus",    "Claude Opus 4.1"),
    ModelInfo("openai/gpt-5",                "gpt5",    "GPT-5"),
    ModelInfo("openai/gpt-4o",               "gpt4o",   "GPT-4o"),
    ModelInfo("google/gemini-2.5-pro",       "gemini",  "Gemini 2.5 Pro"),
    ModelInfo("deepseek/deepseek-chat",      "deepseek", "DeepSeek V3"),
    ModelInfo("x-ai/grok-4",                 "grok",    "Grok 4"),
    ModelInfo("qwen/qwen3.6-plus", "qwen", "QWEN 3 Coder"),
    ModelInfo("openai/gpt-oss-120b", "gpt-oss", "GPT-OSS"),
]

DEFAULT_MODEL: ModelInfo = AVAILABLE_MODELS[-1]


def find_model(query: str) -> ModelInfo | None:
    """Resolve a user-typed model identifier to a ModelInfo.

    Matches on (in order): exact slug, short alias, case-insensitive label.
    Returns None if nothing matches.
    """
    q = query.strip().lower()
    for m in AVAILABLE_MODELS:
        if q == m.slug.lower() or q == m.short.lower() or q == m.label.lower():
            return m
    return None
