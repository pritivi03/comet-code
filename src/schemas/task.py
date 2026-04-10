"""Task-level schemas (mode enum, etc.)."""

from __future__ import annotations

from enum import Enum

from schemas.mode_policy import ModePolicy

class TaskMode(str, Enum):
    EXPLAIN = "explain"
    DEBUG = "debug"
    REFACTOR = "refactor"
    IMPLEMENT = "implement"
    PLAN = "plan"

    @classmethod
    def names(cls) -> list[str]:
        return [m.value for m in cls]

def get_mode_policy_for_task_mode(task_mode: TaskMode) -> ModePolicy:
    match task_mode:
        case TaskMode.EXPLAIN:
            return ModePolicy(
                require_plan_first=False,
                allow_edits=False,
                allow_command_exec=False,
                require_verification=False,
                max_attempts=1,
                expected_output_schema_name="explanation",
            )
        case TaskMode.DEBUG:
            return ModePolicy(
                require_plan_first=False,
                allow_edits=True,
                allow_command_exec=True,
                require_verification=True,
                max_attempts=3,
                expected_output_schema_name="file_edit",
            )
        case TaskMode.REFACTOR:
            return ModePolicy(
                require_plan_first=False,
                allow_edits=True,
                allow_command_exec=True,
                require_verification=True,
                max_attempts=3,
                expected_output_schema_name="file_edit",
            )
        case TaskMode.IMPLEMENT:
            return ModePolicy(
                require_plan_first=False,
                allow_edits=True,
                allow_command_exec=True,
                require_verification=True,
                max_attempts=3,
                expected_output_schema_name="file_edit",
            )
        case TaskMode.PLAN:
            return ModePolicy(
                require_plan_first=True,
                allow_edits=False,
                allow_command_exec=False,
                require_verification=False,
                max_attempts=1,
                expected_output_schema_name="plan",
            )