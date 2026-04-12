from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.nodes import execute_tools_node


def _base_state() -> dict:
    return {
        "messages": [],
        "mode": "implement",
        "model_slug": "fake",
        "max_attempts": 1,
        "tool_style": "native",
        "allow_mutating_tools": True,
        "user_request": "make the change",
        "attempt_number": 0,
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


class ToolApprovalTests(unittest.TestCase):
    def test_read_only_tool_skips_approval(self) -> None:
        state = _base_state()
        state["pending_tool_calls"] = [
            {"id": "tool_0", "tool_name": "read_file", "args": {"path": "src/core/graph.py"}, "reason": None}
        ]

        approval_calls: list[list[dict]] = []

        with patch("core.nodes.execute_tool", return_value="from __future__ import annotations") as execute_mock:
            result = execute_tools_node(
                state,
                request_approval=lambda proposals: approval_calls.append(proposals) or True,
            )

        self.assertEqual(approval_calls, [])
        execute_mock.assert_called_once()
        self.assertEqual(result["tool_calls_used"], 1)

    def test_mutating_tool_requires_approval_and_executes_when_approved(self) -> None:
        state = _base_state()
        state["pending_tool_calls"] = [
            {
                "id": "tool_0",
                "tool_name": "write_file",
                "args": {"path": "tmp/demo.txt", "content": "hello"},
                "reason": "Create the file with the requested content.",
            }
        ]

        seen: list[list[dict]] = []

        with patch("core.nodes.execute_tool", return_value="[ok] wrote tmp/demo.txt") as execute_mock:
            result = execute_tools_node(
                state,
                request_approval=lambda proposals: seen.append(proposals) or True,
            )

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0]["tool_name"], "write_file")
        execute_mock.assert_called_once_with("write_file", {"path": "tmp/demo.txt", "content": "hello"})
        self.assertEqual(result["messages"][0]["content"], "[ok] wrote tmp/demo.txt")

    def test_mutating_tool_rejection_skips_execution_and_returns_tool_message(self) -> None:
        state = _base_state()
        state["pending_tool_calls"] = [
            {
                "id": "tool_0",
                "tool_name": "replace_text",
                "args": {"path": "src/app.py", "old_text": "a", "new_text": "b"},
                "reason": "Update the text.",
            }
        ]

        with patch("core.nodes.execute_tool") as execute_mock:
            result = execute_tools_node(
                state,
                request_approval=lambda proposals: False,
            )

        execute_mock.assert_not_called()
        self.assertEqual(result["messages"][0]["content"], "[rejected] user declined approval for replace_text")
        self.assertIn("user rejected proposed mutation", result["evidence_notes"][0])


if __name__ == "__main__":
    unittest.main()
