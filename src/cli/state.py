"""Mutable shell state — current mode, current model, etc."""

from __future__ import annotations

from dataclasses import dataclass

from llm.models import DEFAULT_MODEL, ModelInfo
from schemas.task import TaskMode


@dataclass
class ShellState:
    """State carried across slash-command invocations within one shell session."""

    mode: TaskMode = TaskMode.IMPLEMENT
    model: ModelInfo = DEFAULT_MODEL
