from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli.ui import _format_proposal_lines


class ApprovalUiTests(unittest.TestCase):
    def test_replace_text_proposal_is_rendered_compactly(self) -> None:
        lines = _format_proposal_lines(
            {
                "tool_name": "replace_text",
                "args": {
                    "path": "src/cli/render.py",
                    "old_text": "class EventRenderer:\n    def __init__(...)",
                    "new_text": "class EventRenderer:\n    def __init__(...)\n        self._live_token_count = 0",
                    "replace_all": False,
                },
                "reason": "Track token count in the renderer.",
            }
        )

        joined = "\n".join(lines)
        self.assertIn("replace_text src/cli/render.py", joined)
        self.assertIn("mode: replace first exact match", joined)
        self.assertIn("why: Track token count in the renderer.", joined)
        self.assertIn("old:", joined)
        self.assertIn("new:", joined)
        self.assertNotIn("old_text=", joined)
        self.assertNotIn("new_text=", joined)

    def test_write_file_proposal_is_rendered_compactly(self) -> None:
        lines = _format_proposal_lines(
            {
                "tool_name": "write_file",
                "args": {
                    "path": "src/new_file.py",
                    "content": "print('hello')\nprint('world')",
                    "create_dirs": True,
                },
                "reason": "Create the new file.",
            }
        )

        joined = "\n".join(lines)
        self.assertIn("write_file src/new_file.py", joined)
        self.assertIn("create_dirs: true", joined)
        self.assertIn("content:", joined)
        self.assertNotIn("content=", joined)


if __name__ == "__main__":
    unittest.main()
