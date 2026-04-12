from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools import build_tool_schema_markdown, execute_tool, tool_requires_approval


class ToolSchemaTests(unittest.TestCase):
    def test_tool_schema_markdown_includes_ranges_and_defaults(self) -> None:
        schema = build_tool_schema_markdown()

        self.assertIn("`read_file`", schema)
        self.assertIn("max_chars: int", schema)
        self.assertIn("default=20000", schema)
        self.assertIn("range=>=200,<=200000", schema)
        self.assertIn("start_line: int range=>=1", schema)
        self.assertIn("`write_file`", schema)
        self.assertIn("Approval: required.", schema)

    def test_read_only_schema_omits_mutating_tools(self) -> None:
        schema = build_tool_schema_markdown(include_mutating=False)

        self.assertNotIn("`write_file`", schema)
        self.assertNotIn("`replace_text`", schema)

    def test_execute_tool_validation_error_is_short_and_actionable(self) -> None:
        result = execute_tool("read_file", {"path": "src/core/graph.py", "max_chars": 100})

        self.assertIn("[error] invalid args for read_file:", result)
        self.assertIn("max_chars", result)
        self.assertIn("greater than or equal to 200", result)
        self.assertNotIn("ValidationError", result)

    def test_tool_requires_approval_for_mutating_tools(self) -> None:
        self.assertTrue(tool_requires_approval("write_file"))
        self.assertTrue(tool_requires_approval("replace_text"))
        self.assertFalse(tool_requires_approval("read_file"))


if __name__ == "__main__":
    unittest.main()
