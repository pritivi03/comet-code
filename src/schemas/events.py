from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    TOKEN = "token"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    LIMIT = "limit"
    ATTEMPT_RETRY = "attempt_retry"
    USAGE = "usage"
    FINAL = "final"
    ERROR = "error"


class StreamEvent(BaseModel):
    type: EventType
    text: str | None = None
    tool_name: str | None = None
    args: dict[str, object] = Field(default_factory=dict)
    output: str | None = None
    error: str | None = None
    reason: str | None = None
    suggestion: str | None = None
    partial_findings: list[str] = Field(default_factory=list)
    failure_kind: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated: bool = True
