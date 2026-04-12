"""LangGraph node functions and conditional routing for the Comet agent."""

from __future__ import annotations

import json
from typing import Any, Callable

from schemas.attempt import ModelResponse, ResponseType
from schemas.events import EventType, StreamEvent
from tools import execute_tool, get_langchain_tools, tool_requires_approval

from core.graph_state import AgentState

MAX_STEPS_PER_ATTEMPT = 18
MAX_TOOL_CALLS_PER_ATTEMPT = 14
MAX_CONSECUTIVE_NO_SIGNAL = 3
MAX_REPEAT_SAME_CALL = 2
ANSWER_RESERVE_STEPS = 1

_EMPTY_FINAL_RETRY_NUDGE = (
    "Your last response contained no text. "
    "Do NOT call any more tools. "
    "Write your complete final answer as plain text right now, "
    "using the evidence you have already gathered. "
    "Start your response with a direct statement about what you found or did."
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
        reason = tc.get("reason")
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
                "reason": reason if isinstance(reason, str) and reason.strip() else None,
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


def _compact_path(text: str) -> str:
    clean = text.strip()
    marker = "/src/"
    if marker in clean:
        return "src/" + clean.split(marker, 1)[1]
    return clean


def _note_from_tool_signal(tool_name: str, output: str) -> str | None:
    first = _first_signal_line(output)
    if not first:
        return None
    compact = _compact_path(first)
    if len(compact) > 140:
        compact = compact[:137].rstrip() + "..."
    return f"{tool_name}: {compact}"


def _one_line_reason(text: str) -> str | None:
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if len(clean) > 160:
            clean = clean[:157].rstrip() + "..."
        return clean
    return None


def _estimate_tokens_text(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_tokens_messages(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for m in messages:
        total_chars += len(str(m.get("role", "")))
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            total_chars += len(json.dumps(content, ensure_ascii=True))
        tool_calls = m.get("tool_calls")
        if tool_calls:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=True))
    if total_chars <= 0:
        return 0
    return max(1, total_chars // 4)


def _build_force_answer_nudge(user_request: str, evidence_notes: list[str]) -> str:
    lines = [
        "Stop using tools and answer the user now with the best response you can.",
        "Do not mention step limits, tool budgets, partial findings, or internal retries.",
        "Use the repo evidence you already gathered.",
        "If uncertainty remains, keep it brief and offer to continue exploring.",
        f"Original request: {user_request}",
    ]
    if evidence_notes:
        lines.append("Relevant repo evidence:")
        lines.extend(f"- {item}" for item in evidence_notes[:5])
    return "\n".join(lines)


def _compose_best_effort_final(state: AgentState) -> str:
    evidence_notes = list(state.get("evidence_notes", []))
    request = (state.get("user_request") or "this").strip()

    lines = ["I haven’t fully finished verifying this yet, but here’s the best answer I can give from what I already checked."]
    if evidence_notes:
        lines.append("")
        lines.append("The most relevant code I found was:")
        lines.extend(f"- {note}" for note in evidence_notes[:4])
        lines.append("")
        lines.append("That should be enough to keep moving, even though I’d still want one more pass to verify the details.")
    else:
        lines.append("")
        lines.append(f"I ran out of useful exploration before I could fully verify {request}.")
    lines.append("")
    lines.append("If you want, I can keep digging and firm this up.")
    return "\n".join(lines)


def _soft_limit_result(
    *,
    failure_kind: str,
    reason: str,
    state: AgentState,
    result_messages: list[dict[str, Any]] | None = None,
    on_event: Callable[[StreamEvent], None] | None = None,
) -> dict[str, Any]:
    evidence_notes = list(state.get("evidence_notes", []))
    _emit_event(
        on_event,
        StreamEvent(
            type=EventType.LIMIT,
            reason=reason,
            failure_kind=failure_kind,
        ),
    )
    return {
        "messages": (result_messages or [])
        + [
            {
                "role": "user",
                "content": _build_force_answer_nudge(
                    user_request=state.get("user_request", ""),
                    evidence_notes=evidence_notes,
                ),
            }
        ],
        "response_type": "retry",
        "attempt_status": None,
        "attempt_failure_reason": reason,
        "failure_reason": reason,
        "failure_kind": failure_kind,
        "force_answer": True,
        "is_done": False,
        "pending_tool_calls": [],
        "evidence_notes": evidence_notes,
    }


def _limit_result(
    reason: str,
    suggestion: str,
    state: AgentState,
    result_messages: list[dict[str, Any]] | None = None,
    on_event: Callable[[StreamEvent], None] | None = None,
) -> dict[str, Any]:
    failure_kind = "insufficient_evidence"
    if "Tool budget exceeded" in reason:
        failure_kind = "budget_exhausted"
    elif "Step limit" in reason:
        failure_kind = "budget_exhausted"
    elif "Repeated tool call pattern" in reason:
        failure_kind = "repeated_low_signal"
    elif "No-signal streak" in reason:
        failure_kind = "insufficient_evidence"
    return _soft_limit_result(
        failure_kind=failure_kind,
        reason=suggestion,
        state=state,
        result_messages=result_messages,
        on_event=on_event,
    )


def _invoke_native(
    llm,
    messages: list[dict[str, Any]],
    on_event: Callable[[StreamEvent], None] | None,
    include_mutating_tools: bool,
) -> tuple[str, list[dict[str, Any]], str]:
    llm_with_tools = llm.bind_tools(get_langchain_tools(include_mutating=include_mutating_tools))

    merged_chunk = None
    streamed_text_parts: list[str] = []
    for chunk in llm_with_tools.stream(messages):
        chunk_text = _to_text(getattr(chunk, "content", ""))
        if chunk_text:
            streamed_text_parts.append(chunk_text)
        if merged_chunk is None:
            merged_chunk = chunk
        else:
            merged_chunk = merged_chunk + chunk

    if merged_chunk is None:
        raise RuntimeError("Model produced no output")

    assistant_text = _to_text(getattr(merged_chunk, "content", ""))
    if not assistant_text.strip() and streamed_text_parts:
        assistant_text = "".join(streamed_text_parts)
    raw_tool_calls = getattr(merged_chunk, "tool_calls", None) or []
    pending_tool_calls = _normalize_tool_calls(raw_tool_calls)
    response_type = ResponseType.TOOL_CALLS.value if pending_tool_calls else ResponseType.FINAL.value

    return assistant_text, pending_tool_calls, response_type


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
            {
                "id": f"json_tool_{idx}",
                "tool_name": tc.tool_name,
                "args": tc.args,
                "reason": tc.reason,
            }
            for idx, tc in enumerate(parsed.tool_calls or [])
        ]
        return raw_content, tool_calls, ResponseType.TOOL_CALLS.value, None

    if parsed.type != ResponseType.FINAL:
        raise RuntimeError(
            f"Fallback adapter expected 'tool_calls' or 'final', got: {parsed.type.value}"
        )

    final_text = parsed.explanation or parsed.summary or ""
    return raw_content, [], ResponseType.FINAL.value, final_text


def _contains_mutating_tool_call(pending_tool_calls: list[dict[str, Any]]) -> bool:
    return any(tool_requires_approval(tc.get("tool_name", "")) for tc in pending_tool_calls)


def make_call_llm_node(
    llm,
    on_event: Callable[[StreamEvent], None] | None = None,
):
    """Returns a node that invokes the model and parses tool calls/final output."""

    def call_llm(state: AgentState) -> dict:
        tool_style = state.get("tool_style", "native")
        force_answer = state.get("force_answer", False)
        pending_tool_calls: list[dict[str, Any]] = []
        final_summary = state.get("final_summary")
        final_explanation = state.get("final_explanation")
        response_type = ResponseType.FINAL.value
        model_response_str = ""
        # Track peak context size (this call's prompt), not a running sum.
        # Accumulating would count the system prompt + history on every LLM call,
        # inflating the reported total by O(n_steps).
        estimated_prompt_tokens = _estimate_tokens_messages(state["messages"])
        estimated_completion_tokens = state.get("estimated_completion_tokens", 0)

        if tool_style == "native":
            assistant_text, pending_tool_calls, response_type = _invoke_native(
                llm=llm,
                messages=state["messages"],
                on_event=on_event,
                include_mutating_tools=state.get("allow_mutating_tools", False),
            )
            model_response_str = assistant_text
            if response_type == ResponseType.TOOL_CALLS.value and force_answer:
                assistant_text = _compose_best_effort_final(state)
                pending_tool_calls = []
                response_type = ResponseType.FINAL.value
                model_response_str = assistant_text
            elif (
                response_type == ResponseType.TOOL_CALLS.value
                and not _contains_mutating_tool_call(pending_tool_calls)
                and state["step_number"] >= MAX_STEPS_PER_ATTEMPT - (ANSWER_RESERVE_STEPS + 1)
            ):
                return _limit_result(
                    reason=f"Step limit ({MAX_STEPS_PER_ATTEMPT}) reached before final answer.",
                    suggestion="I’ve got enough context to give you a best-effort answer now.",
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
                    assistant_text = _compose_best_effort_final(state)
                _emit_event(on_event, StreamEvent(type=EventType.TOKEN, text=assistant_text))
                final_explanation = assistant_text
                final_summary = assistant_text.splitlines()[0] if assistant_text else "Done."
                estimated_completion_tokens += _estimate_tokens_text(assistant_text)
                _emit_event(
                    on_event,
                    StreamEvent(
                        type=EventType.FINAL,
                        text=assistant_text,
                    ),
                )
            else:
                reason = _one_line_reason(assistant_text)
                if reason and pending_tool_calls:
                    pending_tool_calls[0]["reason"] = pending_tool_calls[0].get("reason") or reason
                estimated_completion_tokens += _estimate_tokens_text(assistant_text)
        else:
            raw_content, pending_tool_calls, response_type, final_text = _invoke_json_fallback(
                llm=llm,
                messages=state["messages"],
            )
            model_response_str = raw_content
            if response_type == ResponseType.TOOL_CALLS.value and force_answer:
                final_text = _compose_best_effort_final(state)
                pending_tool_calls = []
                response_type = ResponseType.FINAL.value
                model_response_str = final_text
            elif (
                response_type == ResponseType.TOOL_CALLS.value
                and not _contains_mutating_tool_call(pending_tool_calls)
                and state["step_number"] >= MAX_STEPS_PER_ATTEMPT - (ANSWER_RESERVE_STEPS + 1)
            ):
                return _limit_result(
                    reason=f"Step limit ({MAX_STEPS_PER_ATTEMPT}) reached before final answer.",
                    suggestion="I’ve got enough context to give you a best-effort answer now.",
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
                estimated_completion_tokens += _estimate_tokens_text(final_text or raw_content)
                if final_text:
                    _emit_event(on_event, StreamEvent(type=EventType.TOKEN, text=final_text))
                _emit_event(
                    on_event,
                    StreamEvent(
                        type=EventType.FINAL,
                        text=final_text or raw_content,
                    ),
                )
            else:
                estimated_completion_tokens += _estimate_tokens_text(raw_content)

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
            "failure_kind": None,
            "force_answer": False,
            "step_number": state["step_number"] + 1,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
        }
        if response_type == ResponseType.FINAL.value:
            updates["attempt_status"] = "success"
            updates["is_done"] = True
        return updates

    return call_llm


def execute_tools_node(
    state: AgentState,
    on_event: Callable[[StreamEvent], None] | None = None,
    request_approval: Callable[[list[dict[str, Any]]], bool] | None = None,
) -> dict:
    """Execute pending tool calls and append results as tool messages."""
    result_messages: list[dict[str, Any]] = []
    tool_calls_used = state.get("tool_calls_used", 0)
    consecutive_no_signal = state.get("consecutive_no_signal", 0)
    repeat_call_streak = state.get("repeat_call_streak", 0)
    last_call_fingerprint = state.get("last_call_fingerprint")
    partial_findings = list(state.get("partial_findings", []))
    evidence_notes = list(state.get("evidence_notes", []))

    for tc in state["pending_tool_calls"]:
        if tool_calls_used >= MAX_TOOL_CALLS_PER_ATTEMPT:
            return _limit_result(
                reason=f"Tool budget exceeded ({MAX_TOOL_CALLS_PER_ATTEMPT} calls) in this attempt.",
                suggestion="I’ve already explored enough to give you a best-effort answer.",
                state={**state, "partial_findings": partial_findings, "evidence_notes": evidence_notes},
                result_messages=result_messages,
                on_event=on_event,
            ) | {
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
                "evidence_notes": evidence_notes,
            }

        if tool_requires_approval(tc["tool_name"]):
            if request_approval is None:
                rejection_output = "[error] approval is required for mutating tools, but no approval handler is configured"
                result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tc["tool_name"],
                        "content": rejection_output,
                    }
                )
                return {
                    "messages": result_messages,
                    "pending_tool_calls": [],
                    "tool_calls_used": tool_calls_used,
                    "consecutive_no_signal": consecutive_no_signal,
                    "repeat_call_streak": repeat_call_streak,
                    "last_call_fingerprint": last_call_fingerprint,
                    "partial_findings": partial_findings,
                    "evidence_notes": evidence_notes,
                    "response_type": "call_llm",
                }

            approved = request_approval([tc])
            if not approved:
                rejection_output = f"[rejected] The user declined the proposed change to {tc['tool_name']}. Do not retry this change."
                result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tc["tool_name"],
                        "content": rejection_output,
                    }
                )
                final_explanation = "The proposed change was declined. Let me know if you'd like a different approach."
                _emit_event(on_event, StreamEvent(type=EventType.TOKEN, text=final_explanation))
                _emit_event(on_event, StreamEvent(type=EventType.FINAL, text=final_explanation))
                return {
                    "messages": result_messages,
                    "pending_tool_calls": [],
                    "tool_calls_used": tool_calls_used,
                    "consecutive_no_signal": consecutive_no_signal,
                    "repeat_call_streak": repeat_call_streak,
                    "last_call_fingerprint": last_call_fingerprint,
                    "partial_findings": partial_findings,
                    "evidence_notes": evidence_notes,
                    "attempt_status": "success",
                    "final_explanation": final_explanation,
                    "final_summary": final_explanation,
                    "is_done": True,
                }

        tool_name = tc["tool_name"]
        args: dict[str, Any] = tc.get("args", {})
        reason: str | None = tc.get("reason")
        tool_call_id = tc.get("id", "")
        fingerprint = _call_fingerprint(tool_name, args)

        _emit_event(
            on_event,
            StreamEvent(type=EventType.TOOL_START, tool_name=tool_name, args=args, reason=reason),
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
            note = _note_from_tool_signal(tool_name, output)
            if note and note not in evidence_notes:
                evidence_notes.append(note)
                evidence_notes = evidence_notes[-6:]

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

        # After a mutating tool runs, return immediately so the LLM re-evaluates
        # against the new file state. This prevents duplicate edits when the model
        # batches multiple replace_text calls for the same location.
        if tool_requires_approval(tool_name) and not error:
            return {
                "messages": result_messages,
                "pending_tool_calls": [],
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
                "evidence_notes": evidence_notes,
            }

        if repeat_call_streak >= MAX_REPEAT_SAME_CALL:
            return _limit_result(
                reason=f"Repeated tool call pattern detected ({MAX_REPEAT_SAME_CALL + 1}x).",
                suggestion="I’m going to stop thrashing and answer from the evidence I already have.",
                state={**state, "partial_findings": partial_findings, "evidence_notes": evidence_notes},
                result_messages=result_messages,
                on_event=on_event,
            ) | {
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
                "evidence_notes": evidence_notes,
            }

        if consecutive_no_signal >= MAX_CONSECUTIVE_NO_SIGNAL:
            return _limit_result(
                reason=f"No-signal streak reached limit ({MAX_CONSECUTIVE_NO_SIGNAL}).",
                suggestion="I’m not getting much more signal, so I’ll answer from what I’ve already confirmed.",
                state={**state, "partial_findings": partial_findings, "evidence_notes": evidence_notes},
                result_messages=result_messages,
                on_event=on_event,
            ) | {
                "tool_calls_used": tool_calls_used,
                "consecutive_no_signal": consecutive_no_signal,
                "repeat_call_streak": repeat_call_streak,
                "last_call_fingerprint": last_call_fingerprint,
                "partial_findings": partial_findings,
                "evidence_notes": evidence_notes,
            }

    return {
        "messages": result_messages,
        "pending_tool_calls": [],
        "tool_calls_used": tool_calls_used,
        "consecutive_no_signal": consecutive_no_signal,
        "repeat_call_streak": repeat_call_streak,
        "last_call_fingerprint": last_call_fingerprint,
        "partial_findings": partial_findings,
        "evidence_notes": evidence_notes,
    }


def make_execute_tools_node(
    on_event: Callable[[StreamEvent], None] | None = None,
    request_approval: Callable[[list[dict[str, Any]]], bool] | None = None,
):
    def _node(state: AgentState) -> dict:
        return execute_tools_node(state, on_event=on_event, request_approval=request_approval)

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
