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
    '    {"tool_name": "<name>", "args": {"arg_name": "value"}, "reason": "<optional one-line why>"}\n'
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
    "Budget tool usage. For follow-up edits or narrow requests, inspect the best candidate before doing another broad search.\n"
    "Prefer targeted reads (`read_range`) over full-file reads when you already know the rough location.\n"
    "Do not chain multiple broad searches across the whole repo when one promising file or symbol is already identified.\n"
    "If a tool returns no matches or low-signal output, try another tool call with refined arguments.\n"
    "Do not stop after a single failed tool call when the task is still unresolved.\n"
    "Only provide a final answer once you can cite concrete file/path evidence from tool results.\n"
    "If you have enough evidence to give a best-effort answer, stop exploring and answer.\n"
    "If you are running low on useful exploration, synthesize the best answer you can from the evidence already gathered instead of asking for more tool calls.\n"
    "Before a non-trivial tool call, include a short one-line reason when helpful.\n"
    "Skip the reason for obvious repetitive calls.\n"
    "Never return an empty final response."
)

_MODE_INSTRUCTIONS: dict[TaskMode, str] = {
    TaskMode.EXPLAIN: (
        "Your task is to explain code. "
        "Use read-only tools if needed, then provide a clear explanation. "
        "IMPORTANT: You have NO ability to edit or create files in this mode. "
        "If the user asks you to implement, add, change, or fix something, do NOT claim to have done it. "
        "Instead, explain what would need to change and end with: "
        "\"Switch to implement mode (/mode implement) if you'd like me to make these changes.\""
    ),
    TaskMode.DEBUG: (
        "Your task is to debug and fix the issue. "
        "Use read-only tools to locate and understand the bug, then use write/replace tools to apply the fix."
    ),
    TaskMode.REFACTOR: (
        "Your task is to refactor code. "
        "Use read-only tools to understand the existing code, then use write/replace tools to apply the refactored version."
    ),
    TaskMode.IMPLEMENT: (
        "Your task is to implement a feature or change. "
        "Use read-only tools to understand the codebase, then use write/replace tools to make the necessary edits. "
        "Do not just explain what to do — actually write the code."
    ),
    TaskMode.PLAN: (
        "Your task is to produce a plan. "
        "Use read-only tools if needed, then output a structured step-by-step plan. "
        "IMPORTANT: You have NO ability to edit or create files in this mode. "
        "Do not claim to have made any changes. "
        "If the user wants changes applied, tell them to switch to implement mode (/mode implement)."
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
        include_mutating_tools: bool = False,
    ) -> str:
        tool_schema = "## Available Tools\n" + build_tool_schema_markdown(include_mutating=include_mutating_tools)
        base = _NATIVE_SYSTEM_PREAMBLE if response_style == "native" else _JSON_SYSTEM_PREAMBLE
        parts = [base, tool_schema, _TOOL_EXECUTION_POLICY, _MODE_INSTRUCTIONS[self._mode]]
        if include_mutating_tools:
            parts.append(
                "## Mutation Policy\n"
                "Mutating tools require explicit user approval before they run.\n"
                "Use mutating tools only when needed to complete the task.\n"
                "Prefer precise edits over whole-file rewrites when possible."
            )
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
        include_mutating_tools: bool = False,
    ) -> list[dict[str, str]]:
        system = self.build_system_message(
            response_style=response_style,
            previous_summary=previous_summary,
            failure_context=failure_context,
            include_mutating_tools=include_mutating_tools,
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
