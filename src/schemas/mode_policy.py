from __future__ import annotations

from pydantic import BaseModel

class ModePolicy(BaseModel):
    require_plan_first: bool
    allow_edits: bool
    allow_command_exec: bool
    require_verification: bool
    max_attempts: int
    expected_output_schema_name: str
