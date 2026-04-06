"""Task-level schemas (mode enum, etc.)."""

from __future__ import annotations

from enum import Enum


class TaskMode(str, Enum):
    """High-level operating mode for a task session.

    Mirrors the modes described in PROJECT_SPEC.md. The mode determines
    whether planning is required, whether edits are allowed, and how
    verification + retries behave.
    """

    EXPLAIN = "explain"
    DEBUG = "debug"
    REFACTOR = "refactor"
    IMPLEMENT = "implement"
    PLAN = "plan"

    @classmethod
    def names(cls) -> list[str]:
        return [m.value for m in cls]
