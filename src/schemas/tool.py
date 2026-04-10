from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


class ToolAction(BaseModel):
    tool_name: str
    args: list[str]
    status: ActionStatus = ActionStatus.PROPOSED

    output: str | None = None
    error: str | None = None
