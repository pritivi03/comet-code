"""Mutable shell state — current mode, current model, etc."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm.models import DEFAULT_MODEL, ModelInfo
from schemas.task import TaskMode


@dataclass
class ShellState:
    """State carried across slash-command invocations within one shell session."""

    mode: TaskMode = TaskMode.EXPLAIN
    model: ModelInfo = DEFAULT_MODEL
    tool_view_collapsed: bool = True
    last_tool_history: list[dict] = field(default_factory=list)
