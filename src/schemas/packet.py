from __future__ import annotations

from pydantic import BaseModel

from schemas.code_chunk import CodeChunk

class ModelPacket(BaseModel):
    instructions: str

    selected_chunks: list[CodeChunk]

    failure_context: str | None
    previous_attempt_summary: str | None
    recent_tool_context: str | None

    expected_output_schema_name: str