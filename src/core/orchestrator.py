from __future__ import annotations

import uuid
from typing import Callable

from llm.models import ModelInfo
from llm.openrouter_client import OpenRouterClient
from llm.prompts import PromptBuilder
from schemas.attempt import (
    AttemptRecord,
    AttemptStatus,
    InteractionStep,
    ModelResponse,
    ResponseType,
)
from schemas.session import TaskSession, SessionStatus, SharedContext
from schemas.task import TaskMode, get_mode_policy_for_task_mode

MAX_STEPS_PER_ATTEMPT = 15


def _parse_model_response(raw: str) -> ModelResponse:
    """Parse the raw LLM response string into a ModelResponse."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return ModelResponse.model_validate_json(text.strip())


class Orchestrator:
    def __init__(self, llm_client: OpenRouterClient) -> None:
        self.llm_client = llm_client

    def run_task(
        self,
        user_request: str,
        mode: TaskMode,
        model: ModelInfo,
        on_step: Callable[[InteractionStep], None] | None = None,
    ) -> TaskSession:
        prompt_builder = PromptBuilder(mode)
        policy = get_mode_policy_for_task_mode(mode)

        session = TaskSession(
            session_id=uuid.uuid4().hex,
            repo_root=".",
            user_request=user_request,
            mode=mode,
            status=SessionStatus.RUNNING,
            shared_context=SharedContext(available_tools=[]),
            chunk_store={},
        )

        prev_attempt: AttemptRecord | None = None

        for attempt_num in range(policy.max_attempts):
            attempt = AttemptRecord(
                attempt_number=attempt_num,
                status=AttemptStatus.RUNNING,
            )

            # Seed messages — clean context, with retry info if applicable
            attempt.messages = prompt_builder.build_initial_messages(
                user_request,
                previous_summary=prev_attempt.summary if prev_attempt else None,
                failure_context=prev_attempt.failure_reason if prev_attempt else None,
            )

            for step_num in range(MAX_STEPS_PER_ATTEMPT):
                # Send to LLM and collect full response
                response_buf: list[str] = []
                self.llm_client.fetch(attempt.messages, model, response_buf.append)
                raw_response = "".join(response_buf)

                PromptBuilder.append_assistant_message(attempt.messages, raw_response)

                # Parse structured response
                model_response = _parse_model_response(raw_response)

                step = InteractionStep(
                    step_number=step_num,
                    model_response_str=raw_response,
                    model_response=model_response,
                )

                if model_response.type == ResponseType.TOOL_CALLS and model_response.tool_calls:
                    for tool_action in model_response.tool_calls:
                        # TODO: execute tool, set tool_action.status and .output
                        pass
                    step.tool_actions = model_response.tool_calls
                    for ta in step.tool_actions:
                        PromptBuilder.append_tool_result(
                            attempt.messages, ta.tool_name, ta.output or "",
                        )

                elif model_response.type == ResponseType.EDITS and model_response.edits:
                    attempt.edits.extend(model_response.edits)
                    edit_summary = "\n".join(
                        f"[Edit proposed] {e.file_path} L{e.start_line}-{e.end_line}"
                        for e in model_response.edits
                    )
                    PromptBuilder.append_tool_result(
                        attempt.messages, "edits", edit_summary,
                    )

                elif model_response.type == ResponseType.FINAL:
                    attempt.summary = model_response.summary
                    attempt.status = AttemptStatus.SUCCESS

                attempt.interaction_steps.append(step)

                if on_step:
                    on_step(step)

                # Exit step loop if model is done (edits or final)
                if model_response.type in (ResponseType.EDITS, ResponseType.FINAL):
                    break

            # TODO: if edits were proposed, verify (tests/lint) and set status
            session.attempts.append(attempt)
            prev_attempt = attempt

            if attempt.status == AttemptStatus.SUCCESS:
                break

        session.status = (
            SessionStatus.SUCCESS
            if any(a.status == AttemptStatus.SUCCESS for a in session.attempts)
            else SessionStatus.FAILED
        )
        return session