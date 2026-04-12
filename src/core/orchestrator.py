from __future__ import annotations

from typing import Callable

from core.graph import build_agent_graph
from core.graph_state import AgentState
from llm.models import ModelInfo
from llm.openrouter_client import create_openrouter_llm, supports_native_tool_calling
from llm.prompts import PromptBuilder
from schemas.events import EventType, StreamEvent
from schemas.task import TaskMode, get_mode_policy_for_task_mode


class Orchestrator:
    def __init__(self) -> None:
        self._persistent_state: AgentState | None = None
        self._recent_run_artifacts: list[str] = []

    def _build_recent_artifacts_summary(self) -> str | None:
        if not self._recent_run_artifacts:
            return None
        recent = self._recent_run_artifacts[-3:]
        lines = [f"- {item}" for item in recent]
        return "Recent run artifacts:\n" + "\n".join(lines)

    def _compose_failed_run_response(self, state: AgentState) -> str:
        evidence_notes = list(state.get("evidence_notes", []))
        failure_kind = state.get("failure_kind") or "insufficient_evidence"

        if evidence_notes:
            lines = ["I haven’t finished verifying this yet, but here’s the best answer I can give from what I already checked.", ""]
            lines.append("The most relevant code I found was:")
            lines.extend(f"- {item}" for item in evidence_notes[:4])
            if failure_kind == "repeated_low_signal":
                lines.extend(["", "I stopped there because the search had started looping without adding much new signal."])
            elif failure_kind == "budget_exhausted":
                lines.extend(["", "That should be enough to point the change in the right direction, even though I’d still want one more verification pass."])
            lines.extend(["", "If you want, I can keep exploring and firm this up."])
            return "\n".join(lines)

        return "\n".join(
            [
                "I don’t have a fully verified answer yet, but I can keep exploring from here.",
                "",
                "If you want, I can continue and tighten this up.",
            ]
        )

    def run_task(
        self,
        user_request: str,
        mode: TaskMode,
        model: ModelInfo,
        on_event: Callable[[StreamEvent], None] | None = None,
        request_approval: Callable[[list[dict[str, object]]], bool] | None = None,
    ) -> None:
        policy = get_mode_policy_for_task_mode(mode)
        tool_style = "native" if supports_native_tool_calling(model) else "json"

        llm = create_openrouter_llm(model)
        graph = build_agent_graph(llm=llm, on_event=on_event, request_approval=request_approval)

        new_user_msg: dict = {"role": "user", "content": user_request}
        prior_non_system_messages: list[dict] = []
        mode_changed = False
        if self._persistent_state is not None:
            prior_non_system_messages = [
                m for m in self._persistent_state["messages"]
                if m.get("role") != "system"
            ]
            prev_mode = self._persistent_state.get("mode")
            mode_changed = prev_mode is not None and prev_mode != mode.value

        if mode_changed:
            prior_non_system_messages = [
                *prior_non_system_messages,
                {
                    "role": "user",
                    "content": (
                        f"[mode changed to: {mode.value}] "
                        f"The assistant mode has switched. "
                        f"For this turn and all future turns, operate as a {mode.value} assistant."
                    ),
                },
                {"role": "assistant", "content": f"Understood. Switching to {mode.value} mode."},
            ]

        base_turn_messages = [*prior_non_system_messages, new_user_msg]

        prev_attempt_summary: str | None = self._build_recent_artifacts_summary()
        prev_failure_context: str | None = None
        final_state: AgentState | None = None
        total_estimated_prompt_tokens = 0
        total_estimated_completion_tokens = 0

        for attempt_idx in range(policy.max_attempts):
            system_content = PromptBuilder(mode).build_system_message(
                response_style=tool_style,
                previous_summary=prev_attempt_summary,
                failure_context=prev_failure_context,
                include_mutating_tools=policy.allow_edits,
            )
            attempt_messages = [
                {"role": "system", "content": system_content},
                *base_turn_messages,
            ]
            if attempt_idx > 0 and prev_failure_context:
                attempt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Previous attempt failed. Try a different approach and avoid repeating "
                            "the same low-signal tool calls.\n"
                            f"Failure context: {prev_failure_context}"
                        ),
                    }
                )

            turn_input: AgentState = {
                "messages": attempt_messages,
                "mode": mode.value,
                "model_slug": model.slug,
                "max_attempts": policy.max_attempts,
                "tool_style": tool_style,
                "allow_mutating_tools": policy.allow_edits,
                "user_request": user_request,
                "attempt_number": attempt_idx,
                "attempt_status": None,
                "attempt_failure_reason": None,
                "step_number": 0,
                "response_type": None,
                "pending_tool_calls": [],
                "tool_calls_used": 0,
                "consecutive_no_signal": 0,
                "repeat_call_streak": 0,
                "last_call_fingerprint": None,
                "partial_findings": [],
                "evidence_notes": [],
                "failure_kind": None,
                "force_answer": False,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "final_summary": None,
                "final_explanation": None,
                "failure_reason": None,
                "is_done": False,
            }

            try:
                attempt_state = graph.invoke(turn_input)
            except Exception as exc:
                if on_event:
                    on_event(StreamEvent(type=EventType.ERROR, error=str(exc)))
                raise

            final_state = attempt_state
            total_estimated_prompt_tokens += attempt_state.get("estimated_prompt_tokens", 0)
            total_estimated_completion_tokens += attempt_state.get("estimated_completion_tokens", 0)
            success = attempt_state.get("attempt_status") == "success" and bool(
                (attempt_state.get("final_explanation") or "").strip()
            )
            if success:
                break

            prev_attempt_summary = attempt_state.get("final_summary") or prev_attempt_summary
            prev_failure_context = (
                attempt_state.get("attempt_failure_reason")
                or attempt_state.get("failure_reason")
                or "Attempt ended without a converged answer."
            )
            if on_event and attempt_idx < policy.max_attempts - 1:
                on_event(
                    StreamEvent(
                        type=EventType.ATTEMPT_RETRY,
                        reason=prev_failure_context,
                    )
                )

        if final_state is None:
            raise RuntimeError("No attempt state returned from graph execution.")

        if final_state.get("attempt_status") != "success":
            final_text = self._compose_failed_run_response(final_state)
            final_state["final_explanation"] = final_text
            final_state["final_summary"] = final_text.splitlines()[0] if final_text else "Run ended without a confident final answer."
            if on_event:
                on_event(
                    StreamEvent(
                        type=EventType.FINAL,
                        text=final_text,
                    )
                )

        artifact = final_state.get("final_summary") or final_state.get("attempt_failure_reason")
        if artifact:
            self._recent_run_artifacts.append(artifact)
            self._recent_run_artifacts = self._recent_run_artifacts[-5:]

        if on_event:
            on_event(
                StreamEvent(
                    type=EventType.USAGE,
                    prompt_tokens=total_estimated_prompt_tokens,
                    completion_tokens=total_estimated_completion_tokens,
                    total_tokens=total_estimated_prompt_tokens + total_estimated_completion_tokens,
                    estimated=True,
                )
            )

        self._persistent_state = final_state

    def reset_history(self) -> None:
        self._persistent_state = None
        self._recent_run_artifacts = []
