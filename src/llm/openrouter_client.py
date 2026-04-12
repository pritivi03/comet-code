import os
from typing import Final

from langchain_openai import ChatOpenAI

from llm.models import ModelInfo

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_NATIVE_TOOL_MODEL_PREFIXES: Final[tuple[str, ...]] = (
    "openai/",
    "anthropic/",
    "google/gemini",
    "x-ai/",
    "deepseek/",
)


def create_openrouter_llm(model_info: ModelInfo) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI client configured for OpenRouter."""
    return ChatOpenAI(
        model=model_info.slug,
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url=_OPENROUTER_BASE_URL,
    )

def supports_native_tool_calling(model_info: ModelInfo) -> bool:
    slug = model_info.slug.lower()
    return slug.startswith(_NATIVE_TOOL_MODEL_PREFIXES)
