from __future__ import annotations

import sys
import unittest
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli.render import EventRenderer
from core.nodes import ANSWER_RESERVE_STEPS, MAX_STEPS_PER_ATTEMPT, make_call_llm_node
from core.orchestrator import Orchestrator
from schemas.events import EventType, StreamEvent
from schemas.task import TaskMode


class _FakeChunk:
    def __init__(self, content: str = "", tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def __add__(self, other: "_FakeChunk") -> "_FakeChunk":
        return _FakeChunk(
            content=f"{self.content}{other.content}",
            tool_calls=[*self.tool_calls, *other.tool_calls],
        )


class _FakeNativeLLM:
    def __init__(self, chunk: _FakeChunk) -> None:
        self._chunk = chunk

    def bind_tools(self, _tools) -> "_FakeNativeLLM":
        return self

    def stream(self, _messages):
        yield self._chunk


class LimitHandlingTests(unittest.TestCase):
    def test_limit_event_does_not_render_limit_panel(self) -> None:
        console = Console(record=True, width=100)
        renderer = EventRenderer(console, collapsed_tools=True)

        renderer.render(
            StreamEvent(
                type=EventType.LIMIT,
                reason="I’ve got enough context to answer now.",
                failure_kind="budget_exhausted",
            )
        )

        self.assertEqual(console.export_text(), "")

    def test_force_answer_turn_converts_tool_calls_into_final(self) -> None:
        llm = _FakeNativeLLM(
            _FakeChunk(
                content="I want more tools",
                tool_calls=[
                    {
                        "id": "tool_0",
                        "name": "read_file",
                        "args": {"path": "src/core/nodes.py"},
                    }
                ],
            )
        )
        node = make_call_llm_node(llm)

        result = node(
            {
                "messages": [{"role": "user", "content": "do the same thing for tool calls"}],
                "mode": TaskMode.IMPLEMENT.value,
                "model_slug": "fake",
                "max_attempts": 1,
                "tool_style": "native",
                "user_request": "do the same thing for tool calls",
                "attempt_number": 0,
                "attempt_status": None,
                "attempt_failure_reason": "I’ve already explored enough to answer.",
                "step_number": 3,
                "response_type": None,
                "pending_tool_calls": [],
                "tool_calls_used": 6,
                "consecutive_no_signal": 0,
                "repeat_call_streak": 0,
                "last_call_fingerprint": None,
                "partial_findings": [],
                "evidence_notes": ["search_text: src/core/nodes.py:360:def execute_tools_node("],
                "failure_kind": "budget_exhausted",
                "force_answer": True,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "final_summary": None,
                "final_explanation": None,
                "failure_reason": None,
                "is_done": False,
            }
        )

        self.assertEqual(result["response_type"], "final")
        self.assertTrue(result["is_done"])
        self.assertIn("best answer I can give", result["final_explanation"])
        self.assertEqual(result["pending_tool_calls"], [])

    def test_penultimate_tool_request_triggers_soft_limit_retry(self) -> None:
        llm = _FakeNativeLLM(
            _FakeChunk(
                content="Need one more read",
                tool_calls=[
                    {
                        "id": "tool_0",
                        "name": "read_range",
                        "args": {"path": "src/cli/render.py", "start_line": 1, "end_line": 50},
                    }
                ],
            )
        )
        node = make_call_llm_node(llm)

        result = node(
            {
                "messages": [{"role": "user", "content": "do the same thing for tool calls"}],
                "mode": TaskMode.IMPLEMENT.value,
                "model_slug": "fake",
                "max_attempts": 1,
                "tool_style": "native",
                "user_request": "do the same thing for tool calls",
                "attempt_number": 0,
                "attempt_status": None,
                "attempt_failure_reason": None,
                "step_number": MAX_STEPS_PER_ATTEMPT - (ANSWER_RESERVE_STEPS + 1),
                "response_type": None,
                "pending_tool_calls": [],
                "tool_calls_used": 5,
                "consecutive_no_signal": 0,
                "repeat_call_streak": 0,
                "last_call_fingerprint": None,
                "partial_findings": [],
                "evidence_notes": ["read_range: src/cli/render.py:148: def _render_tool_end(self, event: StreamEvent) -> None:"],
                "failure_kind": None,
                "force_answer": False,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "final_summary": None,
                "final_explanation": None,
                "failure_reason": None,
                "is_done": False,
            }
        )

        self.assertEqual(result["response_type"], "retry")
        self.assertTrue(result["force_answer"])
        self.assertEqual(result["failure_kind"], "budget_exhausted")
        self.assertIn("Stop using tools and answer the user now", result["messages"][-1]["content"])

    def test_mutating_tool_request_bypasses_soft_limit_retry(self) -> None:
        llm = _FakeNativeLLM(
            _FakeChunk(
                content="Apply the edit now",
                tool_calls=[
                    {
                        "id": "tool_0",
                        "name": "replace_text",
                        "args": {"path": "src/cli/render.py", "old_text": "a", "new_text": "b"},
                    }
                ],
            )
        )
        node = make_call_llm_node(llm)

        result = node(
            {
                "messages": [{"role": "user", "content": "update the renderer"}],
                "mode": TaskMode.IMPLEMENT.value,
                "model_slug": "fake",
                "max_attempts": 1,
                "tool_style": "native",
                "allow_mutating_tools": True,
                "user_request": "update the renderer",
                "attempt_number": 0,
                "attempt_status": None,
                "attempt_failure_reason": None,
                "step_number": MAX_STEPS_PER_ATTEMPT - (ANSWER_RESERVE_STEPS + 1),
                "response_type": None,
                "pending_tool_calls": [],
                "tool_calls_used": 5,
                "consecutive_no_signal": 0,
                "repeat_call_streak": 0,
                "last_call_fingerprint": None,
                "partial_findings": [],
                "evidence_notes": ["read_range: src/cli/render.py:10: class EventRenderer:"],
                "failure_kind": None,
                "force_answer": False,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "final_summary": None,
                "final_explanation": None,
                "failure_reason": None,
                "is_done": False,
            }
        )

        self.assertEqual(result["response_type"], "tool_calls")
        self.assertEqual(result["pending_tool_calls"][0]["tool_name"], "replace_text")

    def test_failed_run_response_avoids_internal_limit_jargon(self) -> None:
        orchestrator = Orchestrator()
        text = orchestrator._compose_failed_run_response(
            {
                "evidence_notes": ["read_range: src/core/nodes.py:360:def execute_tools_node("],
                "failure_kind": "budget_exhausted",
            }
        )

        self.assertNotIn("Step limit", text)
        self.assertNotIn("Tool budget", text)
        self.assertNotIn("Partial findings", text)
        self.assertIn("If you want, I can keep exploring", text)


if __name__ == "__main__":
    unittest.main()
