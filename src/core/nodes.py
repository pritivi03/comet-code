"""LangGraph node functions and conditional routing for the Comet agent."""

from __future__ import annotations

import json
from typing import Any, Callable

from schemas.attempt import ModelResponse, ResponseType
from schemas.events import EventType, StreamEvent
from tools import execute_tool, get_langchain_tools

from core.graph_state import AgentState

MAX_STEPS_PER_ATTEMPT = 15
MAX_TOOL_CALLS_PER_ATTEMPT = 12
MAX_CONSECUTIVE_NO_SIGNAL = 3
MAX_REPEAT_SAME_CALL = 2

_EMPTY_FINAL_RETRY_NUDGE = (
    "Your previous response did not provide a usable final answer. "
    "Continue using tools to gather evidence and then answer the user clearly."
)


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "".join(parts)
    return ""


def _normalize_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, tc in enumerate(raw_tool_calls):
        tool_name = tc.get("name")
        raw_args = tc.get("args", {})
        tool_call_id = tc.get("id") or f"tool_{idx}"
        if not isinstance(tool_name, str) or not tool_name:
            continue

        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {}

        normalized.append(
            {
                "id": tool_call_id,
                "tool_name": tool_name,
                "args": parsed_args,
            }
        )
    return normalized


def _emit_event(
    on_event: Callable[[StreamEvent], None] | None,
    event: StreamEvent,
) -> None:
    if on_event:
        on_event(event)


def _call_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=True)}"


def _is_no_signal_output(output: str) -> bool:
    text = output.strip().lower()
    return (
        text.startswith("[no matches]")
        or text.startswith("[no files found]")
        or text.startswith("[error]")
    )


def _first_signal_line(output: str) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _limit_result(
    reason: str,
    suggestion: str,
    state: AgentState,
    result_messages: list[dict[str, Any]] | None = None,
    on_event: Callable[[StreamEvent], None] | None = None,
) -> dict[str, Any]:
    _emit_event(
        on_event,
        StreamEvent(
            type=EventType.LIMIT,
            reason=reason,
            suggestion=suggestion,
            partial_findings=state.get("partial_findings", []),
        ),
    )
    return {
        "messages": result_messages or [],
        "response_type": "attempt_failed",
        "attempt_status": "failed",
        "attempt_failure_reason": reason,
        "failure_reason": reason,
        "is_done": True,
        "pending_tool_calls": [],
    }


def _invoke_native(
    llm,
    messages: list[dict[str, Any]],
    on_event: Callable[[StreamEvent], None] | None,
) -> tuple[str, list[dict[str, Any]], str, bool]:
    llm_with_tools = llm.bind_tools(get_langchain_tools())

    merged_chunk = None
    saw_text_token = False
    for chunk in llm_with_tools.stream(messages):
        if merged_chunk is None:
            merged_chunk = chunk
        else:
            merged_chunk = merged_chunk + chunk

        text = _to_text(getattr(chunk, "content", ""))
        if text:
            saw_text_token = True
            _emit_event(on_event, StreamEvent(type=EventType.TOKEN, text=text))

    if merged_chunk is None:
        raise RuntimeError("Model produced no output")

    assistant_text = _to_text(getattr(merged_chunk, "content", ""))
    raw_tool_calls = getattr(merged_chunk, "tool_calls", None) or []
    pending_tool_calls = _normalize_tool_calls(raw_tool_calls)
    response_type = ResponseType.TOOL_CALLS.value if pending_tool_calls else ResponseType.FINAL.value

    return assistant_text, pending_tool_calls, response_type, saw_text_token


def _invoke_json_fallback(
    llm,
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str, str | None]:
    structured_llm = llm.with_structured_output(ModelResponse, include_raw=True)
    result = structured_llm.invoke(messages)

    raw_msg = result["raw"]
    parsed: ModelResponse | None = result["parsed"]
    parse_error = result.get("parsing_error")
    raw_content = _to_text(getattr(raw_msg, "content", ""))

    if parsed is None:
        raise RuntimeError(
            "Fallback JSON parsing failed. "
            f"Parse error: {parse_error}. Raw content: {raw_content[:300]!r}"
        )

    if parsed.type == ResponseType.TOOL_CALLS:
        tool_calls = [
            {"id": f"json_tool_{idx}", "tool_name": tc.tool_name, "args": tc.args}
            for idx, tc in enumerate(parsed.tool_calls or [])
        ]
        return raw_content, tool_calls, ResponseType.TOOL_CALLS.value, None

    if parsed.type != ResponseType.FINAL:
        raise RuntimeError(
            f"Fallback adapter expected 'tool_calls' or 'final', got: {parsed.type.value}"
        )

    final_text = parsed.explanation or parsed.summary or ""
    return raw_content, [], ResponseType.FINAL.value, final_text


def make_call_llm_node(
    llm,
    on_event: Callable[[StreamEvent], None] | None = None,
):
    """Returns a node that invokes the model and parses tool calls/final output."""

    def call_llm(state: AgentState) -> dict:
        tool_style = state.get("tool_style", "native")
        pending_tool_calls: list[dict[str, Any]] = []
        final_summary = state.get("final_summary")
        final_explanation = state.get("final_explanation")
        response_type = ResponseType.FINAL.value
        model_response_str = ""

        if tool_style == "native":
            assistant_text, pending_tool_calls, response_type, saw_text_token = _invoke_native(
                llm=llm,
                messages=state["messages"],
                on_event=on_event,
            )
            model_response_str = assistant_text
            if response_type == ResponseType.TOOL_CALLS.value and state["step_number"] >= MAX_STEPS_PER_ATTEMPT - 1:
                return _limit_result(
                    reason=f"Step limit ({MAX_STEPS_PER_ATTEMPT}) reached before final answer.",
                    suggestion="Narrow the query or ask for a specific file/symbol first.",
                    state=state,
                    on_event=on_event,
                ) | {"step_number": state["step_number"] + 1}

            if response_type == ResponseType.FINAL.value:
                if not assistant_text.strip():
                    if state["step_number"] < MAX_STEPS_PER_ATTEMPT - 1:
                        return {
                            "messages": [{"role": "user", "content": _EMPTY_FINAL_RETRY_NUDGE}],
                            "response_type": "retry",
                            "pending_tool_calls": [],
                            "failure_reason": None,
                            "step_number": state["step_number"] + 1,
                        }
                    assistant_text = (
                        "I could not produce a complete final explanation from this model response. "
                        "Please retry or switch models."
                    )
                if not saw_text_token:
                    _emit_event(on_event, StreamEvent(type=EventType.TOKEN, text=assistant_text))
                final_explanation = assistant_text
                final_summary = assistant_text.splitlines()[0] if assistant_text else "Done."
                _emit_event(
                    on_event,
                    StreamEvent(
                        type=EventType.FINAL,
                        text=assistant_text,
                    ),
                )
        else:
            raw_content, pending_tool_calls, response_type, final_text = _invoke_json_fallback(
                llm=llm,
                messages=state["messages"],
            )
            model_response_str = raw_content
            if response_type == ResponseType.TOOL_CALLS.value and state["step_number"] >= MAX_STEPS_PER_ATTEMPT - 1:
                return _limit_result(
                    reason=f"Step limit ({MAX_STEPS_PER_ATTEMPT}) reached before final answer.",
                    suggestion="Narrow the query or ask for a specific file/symbol first.",
                    state=state,
                    on_event=on_event,
                ) | {"step_number": state["step_number"] + 1}
            if response_type == ResponseType.FINAL.value:
                final_explanation = final_text or raw_content
                if not (final_explanation or "").strip() and state["step_number"] < MAX_STEPS_PER_ATTEMPT - 1:
                    return {
                        "messages": [{"role": "user", "content": _EMPTY_FINAL_RETRY_NUDGE}],
                        "response_type": "retry",
                        "pending_tool_calls": [],
                        "failure_reason": None,
                        "step_number": state["step_number"] + 1,
                    }
                final_summary = (final_text or raw_content).splitlines()[0] if (final_text or raw_content) else "Done."
                if final_text:
                    _emit_event(on_event, StreamEvent(type=EventType.TOKEN, text=final_text))
                _emit_event(
                    on_event,
                    StreamEvent(
                        type=EventType.FINAL,
                        text=final_text or raw_content,
                    ),
                )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": model_response_str,
        }
        if pending_tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["tool_name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in pending_tool_calls
            ]

        updates: dict[str, Any] = {
            "messages": [assistant_message],
            "response_type": response_type,
            "pending_tool_calls": pending_tool_calls,
            "final_summary": final_summary,
            "final_explanation": final_explanation,
            "failure_reason": None,
            "step_number": state["step_number"] + 1,
        }
        if response_type == ResponseType.FINAL.value:
            updates["attempt_status"] = "success"
            updates["is_done"] = True
        return updates

    return call_llm


def execute_tools_node(
    state: AgentState,
    on_event: Callable[[StreamEvent], None] | None = None,
) -> dict:
    """Execute pending tool calls and append results as tool messages."""
    result_messages: list[dict[str, Any]] = []
    tool_calls_used = state.get("tool_calls_used", 0)
    consecutive_no_signal = state.get("consecutive_no_signal", 0)
    repeat_call_streak = state.get("repeat_call_streak", 0)
    last_call_fingerprint = state.get("last_call_fingerprint")
    partial_findings = list(state.get("partial_findings", []))

    for tc in state["pending_tool_calls"]:
        if tool_calls_used >= MAX_TOOL_CALLS_PER_ATTEMPT:
            return _limit_result(
                reason=f"Tool budget exceeded ({MAX_TOOL_CALLS_PER_ATTEMPT} calls) in this attempt.",
                suggestion="Ask a narrower question (specific file/symbol) to reduce search breadth.",
                state={**state, "partial_findings": partial_findings},
                result_messages=result_messages,
                on_event=on_event,
            ) | {
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
            }

        tool_name = tc["tool_name"]
        args: dict[str, Any] = tc.get("args", {})
        tool_call_id = tc.get("id", "")
        fingerprint = _call_fingerprint(tool_name, args)

        _emit_event(
            on_event,
            StreamEvent(type=EventType.TOOL_START, tool_name=tool_name, args=args),
        )

        output = execute_tool(tool_name, args)
        tool_calls_used += 1

        if fingerprint == last_call_fingerprint:
            repeat_call_streak += 1
        else:
            repeat_call_streak = 0
        last_call_fingerprint = fingerprint

        no_signal = _is_no_signal_output(output)
        if no_signal:
            consecutive_no_signal += 1
        else:
            consecutive_no_signal = 0
            finding = _first_signal_line(output)
            if finding:
                partial_findings.append(f"{tool_name}: {finding}")
                partial_findings = partial_findings[-8:]

        error = output if output.startswith("[error]") else None
        _emit_event(
            on_event,
            StreamEvent(
                type=EventType.TOOL_END,
                tool_name=tool_name,
                args=args,
                output=output if not error else None,
                error=error,
            ),
        )

        result_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": output,
            }
        )

        if repeat_call_streak >= MAX_REPEAT_SAME_CALL:
            return _limit_result(
                reason=f"Repeated tool call pattern detected ({MAX_REPEAT_SAME_CALL + 1}x).",
                suggestion="Try a different approach: locate target file first, then inspect direct references.",
                state={**state, "partial_findings": partial_findings},
                result_messages=result_messages,
                on_event=on_event,
            ) | {
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
            }

        if consecutive_no_signal >= MAX_CONSECUTIVE_NO_SIGNAL:
            return _limit_result(
                reason=f"No-signal streak reached limit ({MAX_CONSECUTIVE_NO_SIGNAL}).",
                suggestion="Refine the query with exact filename/import path or ask for one directory scope.",
                state={**state, "partial_findings": partial_findings},
                result_messages=result_messages,
                on_event=on_event,
            ) | {
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
            }

    return {
        "messages": result_messages,
        "pending_tool_calls": [],
        "tool_calls_used": tool_calls_used,
        "consecutive_no_signal": consecutive_no_signal,
        "repeat_call_streak": repeat_call_streak,
        "last_call_fingerprint": last_call_fingerprint,
        "partial_findings": partial_findings,
    }


def make_execute_tools_node(
    on_event: Callable[[StreamEvent], None] | None = None,
):
    def _node(state: AgentState) -> dict:
        return execute_tools_node(state, on_event=on_event)

    return _node


def route_on_response_type(state: AgentState) -> str:
    """Route after call_llm based on the response type and safety guards."""
    if state.get("response_type") == "attempt_failed" or state.get("is_done"):
        return "end"
    if state["step_number"] >= MAX_STEPS_PER_ATTEMPT:
        return "end"

    response_type = state.get("response_type")
    if response_type == ResponseType.TOOL_CALLS.value:
        return "execute_tools"
    if response_type == "retry":
        return "retry"
    return "end"


def route_after_tools(state: AgentState) -> str:
    if state.get("response_type") == "attempt_failed" or state.get("is_done"):
        return "end"
    return "call_llm"
