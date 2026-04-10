"""Prompt construction for LLM calls."""

from __future__ import annotations

from schemas.task import TaskMode


_SYSTEM_PREAMBLE = (
    "You are Comet, an autonomous coding assistant. "
    "You operate on a real codebase using tools. "
    "Be precise, minimal, and grounded — never guess at file contents or repo state."
)

_MODE_INSTRUCTIONS: dict[TaskMode, str] = {
    TaskMode.EXPLAIN: (
        "Your task is to explain code. "
        "Do not propose edits or run commands. "
        "Provide a clear, structured explanation of how the relevant code works."
    ),
    TaskMode.DEBUG: (
        "Your task is to find and fix a bug. "
        "Use tools to investigate, identify the root cause, and propose minimal edits to fix it. "
        "After editing, verification will run automatically."
    ),
    TaskMode.REFACTOR: (
        "Your task is to refactor code. "
        "Preserve existing behavior while improving structure, clarity, or performance. "
        "Use tools to understand the code before proposing edits."
    ),
    TaskMode.IMPLEMENT: (
        "Your task is to implement a feature or change. "
        "Use tools to understand the existing codebase, then propose edits. "
        "Keep changes focused on what was requested."
    ),
    TaskMode.PLAN: (
        "Your task is to produce a plan. "
        "Do not propose edits or run commands. "
        "Analyze the request and output a structured, step-by-step plan."
    ),
}


class PromptBuilder:
    """Builds the message list for an LLM call.

    Currently, handles the initial prompt only: system preamble + mode
    instructions + user request. Will be extended later to include
    project cache content, tool descriptions, and multi-turn context.
    """

    def __init__(self, mode: TaskMode) -> None:
        self._mode = mode

    def build_system_message(
        self,
        previous_summary: str | None = None,
        failure_context: str | None = None,
    ) -> str:
        parts = [_SYSTEM_PREAMBLE, _MODE_INSTRUCTIONS[self._mode]]

        if previous_summary or failure_context:
            retry_section = "\n## Previous Attempt"
            if previous_summary:
                retry_section += f"\nSummary: {previous_summary}"
            if failure_context:
                retry_section += f"\nFailure: {failure_context}"
            parts.append(retry_section)

        return "\n\n".join(parts)

    def build_initial_messages(
        self,
        user_request: str,
        previous_summary: str | None = None,
        failure_context: str | None = None,
    ) -> list[dict[str, str]]:
        system = self.build_system_message(previous_summary, failure_context)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_request},
        ]

    @staticmethod
    def append_assistant_message(messages: list[dict[str, str]], content: str) -> None:
        messages.append({"role": "assistant", "content": content})

    @staticmethod
    def append_tool_result(messages: list[dict[str, str]], tool_name: str, output: str) -> None:
        messages.append({"role": "user", "content": f"[Tool: {tool_name}]\n{output}"})
