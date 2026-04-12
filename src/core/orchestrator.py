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

    def run_task(
        self,
        user_request: str,
        mode: TaskMode,
        model: ModelInfo,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> None:
        policy = get_mode_policy_for_task_mode(mode)
        tool_style = "native" if supports_native_tool_calling(model) else "json"

        llm = create_openrouter_llm(model)
        graph = build_agent_graph(llm=llm, on_event=on_event)

        new_user_msg: dict = {"role": "user", "content": user_request}
        prior_non_system_messages: list[dict] = []
        if self._persistent_state is not None:
            prior_non_system_messages = [
                m for m in self._persistent_state["messages"]
                if m.get("role") != "system"
            ]

        base_turn_messages = [*prior_non_system_messages, new_user_msg]

        prev_attempt_summary: str | None = self._build_recent_artifacts_summary()
        prev_failure_context: str | None = None
        final_state: AgentState | None = None

        for attempt_idx in range(policy.max_attempts):
            system_content = PromptBuilder(mode).build_system_message(
                response_style=tool_style,
                previous_summary=prev_attempt_summary,
                failure_context=prev_failure_context,
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
            partial_findings = final_state.get("partial_findings", [])
            reason = (
                final_state.get("attempt_failure_reason")
                or final_state.get("failure_reason")
                or "Could not converge to a final answer."
            )
            lines = [
                "I couldn’t fully solve this within the current limits.",
                "",
                f"Reason: {reason}",
            ]
            if partial_findings:
                lines.extend(["", "Partial findings:", *[f"- {line}" for line in partial_findings[:5]]])
            lines.extend(
                [
                    "",
                    "Next best step:",
                    "- Ask a narrower query (specific file/import/symbol) and I’ll continue from there.",
                ]
            )
            final_text = "\n".join(lines)
            final_state["final_explanation"] = final_text
            final_state["final_summary"] = "Run exhausted limits before converging."
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

        self._persistent_state = final_state

    def reset_history(self) -> None:
        self._persistent_state = None
        self._recent_run_artifacts = []
