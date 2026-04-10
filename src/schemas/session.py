from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from schemas.attempt import AttemptRecord
from schemas.code_chunk import CodeChunk
from schemas.task import TaskMode


class SessionStatus(str, Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class SharedContext(BaseModel):
    available_tools: list[str]
    project_cache_path: str | None = None


class TaskSession(BaseModel):
    session_id: str
    repo_root: str
    user_request: str
    mode: TaskMode
    status: SessionStatus

    shared_context: SharedContext
    chunk_store: dict[str, CodeChunk]

    attempts: list[AttemptRecord] = []

    final_summary: str | None = None
    final_diff: str | None = None
