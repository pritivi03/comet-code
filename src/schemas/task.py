"""Task-level schemas (mode enum, etc.)."""

from __future__ import annotations

from enum import Enum


class TaskMode(str, Enum):
    EXPLAIN = "explain"
    DEBUG = "debug"
    REFACTOR = "refactor"
    IMPLEMENT = "implement"
    PLAN = "plan"

    @classmethod
    def names(cls) -> list[str]:
        return [m.value for m in cls]
