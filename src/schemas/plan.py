from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum

class PlanItemType(str, Enum):
    SEARCH = "SEARCH"
    INSPECT = "INSPECT"
    EDIT = "EDIT"
    TEST = "TEST"
    VERIFY = "VERIFY"
    EXPLAIN = "EXPLAIN"

class PlanItemStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"

class PlanItem(BaseModel):
    item_id: str
    title: str
    description: str
    type: PlanItemType
    status: PlanItemStatus

    related_files: list[str] = Field(default_factory=list)
    notes: str | None = None

class Plan(BaseModel):
    summary: str
    planning_rationale: str
    items: list[PlanItem]
