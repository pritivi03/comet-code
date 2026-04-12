from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from schemas.plan import Plan
from schemas.tool import ActionStatus, ToolAction


class AttemptStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCESS = "SUCCESS"


class FileEdit(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    original: str
    replacement: str

    status: ActionStatus = ActionStatus.PROPOSED
    error: str | None = None


class ResponseType(str, Enum):
    TOOL_CALLS = "tool_calls"
    EDITS = "edits"
    FINAL = "final"


class ModelResponse(BaseModel):
    type: ResponseType

    tool_calls: list[ToolAction] | None = None
    edits: list[FileEdit] | None = None

    summary: str | None = None
    explanation: str | None = None


class InteractionStep(BaseModel):
    step_number: int

    model_response_str: str
    model_response: ModelResponse

    tool_actions: list[ToolAction] = Field(default_factory=list)


class AttemptRecord(BaseModel):
    attempt_number: int
    status: AttemptStatus

    plan: Plan | None = None

    messages: list[dict[str, str]] = Field(default_factory=list)
    interaction_steps: list[InteractionStep] = Field(default_factory=list)
    edits: list[FileEdit] = Field(default_factory=list)

    summary: str | None = None
    failure_reason: str | None = None
