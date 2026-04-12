"""LangGraph shared state for the Comet agent."""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict


def _append_messages(left: list[dict], right: list[dict]) -> list[dict]:
    """Simple append reducer for OpenAI-format message dicts."""
    return left + right


class AgentState(TypedDict):
    # ------------------------------------------------------------------ #
    # Persists across shell turns — the conversation history fix          #
    # ------------------------------------------------------------------ #
    messages: Annotated[list[dict], _append_messages]

    # ------------------------------------------------------------------ #
    # Set at the start of each shell turn                                 #
    # ------------------------------------------------------------------ #
    mode: str          # TaskMode.value
    model_slug: str    # ModelInfo.slug
    max_attempts: int  # from ModePolicy
    tool_style: str    # "native" | "json"

    # ------------------------------------------------------------------ #
    # Written by nodes during a turn                                      #
    # ------------------------------------------------------------------ #
    attempt_number: int
    attempt_status: str | None       # "success" | "failed"
    attempt_failure_reason: str | None
    step_number: int
    response_type: str | None        # "tool_calls" | "final" | "retry"
    pending_tool_calls: list[dict]   # ToolAction dicts awaiting execution
    tool_calls_used: int
    consecutive_no_signal: int
    repeat_call_streak: int
    last_call_fingerprint: str | None
    partial_findings: list[str]
    final_summary: str | None
    final_explanation: str | None
    failure_reason: str | None
    is_done: bool
