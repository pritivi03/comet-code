"""Prompt construction for LLM calls."""

from __future__ import annotations

from typing import Literal

from schemas.task import TaskMode
from tools import build_tool_schema_markdown


_NATIVE_SYSTEM_PREAMBLE = (
    "You are Comet, an autonomous coding assistant. "
    "You operate on a real codebase using tools. "
    "Be precise, minimal, and grounded — never guess at file contents or repo state.\n"
    "\n"
    "Use tools whenever codebase inspection is needed. "
    "Do not invent file contents."
)

_JSON_SYSTEM_PREAMBLE = (
    "You are Comet, an autonomous coding assistant. "
    "You operate on a real codebase using tools. "
    "Be precise, minimal, and grounded — never guess at file contents or repo state.\n"
    "\n"
    "## Response Format\n"
    "\n"
    "You MUST respond with valid JSON matching one of these two response types.\n"
    "Every response must have a \"type\" field.\n"
    "\n"
    "### type: \"tool_calls\"\n"
    "Use tools to inspect the codebase before acting. You may call multiple tools at once.\n"
    "```json\n"
    "{\n"
    '  "type": "tool_calls",\n'
    '  "tool_calls": [\n'
    '    {"tool_name": "<name>", "args": {"arg_name": "value"}}\n'
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    "### type: \"final\"\n"
    "Use when you are done — to provide an explanation, summary, or both.\n"
    "```json\n"
    "{\n"
    '  "type": "final",\n'
    '  "summary": "<short summary of what was done>",\n'
    '  "explanation": "<detailed explanation if applicable, otherwise null>"\n'
    "}\n"
    "```\n"
    "\n"
    "Do NOT include any text outside the JSON object. Respond with raw JSON only."
)

_TOOL_EXECUTION_POLICY = (
    "## Tool Execution Policy\n"
    "Keep using tools until you have enough concrete repo evidence to answer the user's request.\n"
    "When locating files, prefer filename-oriented tools (list/find) before content search.\n"
    "Use literal text search first; only use regex search when explicitly needed.\n"
    "If a tool returns no matches or low-signal output, try another tool call with refined arguments.\n"
    "Do not stop after a single failed tool call when the task is still unresolved.\n"
    "Only provide a final answer once you can cite concrete file/path evidence from tool results.\n"
    "Never return an empty final response."
)

_MODE_INSTRUCTIONS: dict[TaskMode, str] = {
    TaskMode.EXPLAIN: (
        "Your task is to explain code. "
        "Use read-only tools if needed, then provide a clear explanation."
    ),
    TaskMode.DEBUG: (
        "Your task is to debug. "
        "In this stage, only repository exploration tools are available."
    ),
    TaskMode.REFACTOR: (
        "Your task is to refactor. "
        "In this stage, only repository exploration tools are available."
    ),
    TaskMode.IMPLEMENT: (
        "Your task is to implement a feature or change. "
        "In this stage, only repository exploration tools are available."
    ),
    TaskMode.PLAN: (
        "Your task is to produce a plan. "
        "Use read-only tools if needed, then output a structured step-by-step plan."
    ),
}

_CONCISE_RESPONSE_INSTRUCTION = (
    "## Response Style\n"
    "For modes other than plan, keep responses concise and high-signal. "
    "Default to 1-3 short paragraphs. Use bullets only when they improve clarity."
)


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
        response_style: Literal["native", "json"] = "native",
        previous_summary: str | None = None,
        failure_context: str | None = None,
    ) -> str:
        tool_schema = "## Available Tools\n" + build_tool_schema_markdown()
        base = _NATIVE_SYSTEM_PREAMBLE if response_style == "native" else _JSON_SYSTEM_PREAMBLE
        parts = [base, tool_schema, _TOOL_EXECUTION_POLICY, _MODE_INSTRUCTIONS[self._mode]]
        if self._mode != TaskMode.PLAN:
            parts.append(_CONCISE_RESPONSE_INSTRUCTION)

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
        response_style: Literal["native", "json"] = "native",
        previous_summary: str | None = None,
        failure_context: str | None = None,
    ) -> list[dict[str, str]]:
        system = self.build_system_message(
            response_style=response_style,
            previous_summary=previous_summary,
            failure_context=failure_context,
        )
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
