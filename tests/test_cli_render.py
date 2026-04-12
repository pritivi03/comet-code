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
from schemas.events import EventType, StreamEvent


class EventRendererTests(unittest.TestCase):
    def _make_renderer(self, *, collapsed_tools: bool = True) -> tuple[Console, EventRenderer]:
        console = Console(record=True, width=100)
        return console, EventRenderer(console, collapsed_tools=collapsed_tools)

    def test_final_renders_condensed_tool_history_before_response(self) -> None:
        console, renderer = self._make_renderer()

        renderer.render(
            StreamEvent(
                type=EventType.TOOL_START,
                tool_name="read_file",
                args={"path": "src/cli/render.py"},
            )
        )
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="read_file", output="alpha\nbeta"))
        renderer.render(StreamEvent(type=EventType.FINAL, text="Done"))

        output = console.export_text()
        self.assertIn("tool", output)
        self.assertIn("Ran read_file path=src/cli/render.py done", output)
        self.assertIn("└ alpha", output)
        self.assertLess(output.index("tool"), output.index("response"))

    def test_multiple_tools_preserve_sequence_in_final_summary(self) -> None:
        console, renderer = self._make_renderer()

        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="list_files", args={"path": "src"}))
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="list_files", output="src"))
        renderer.render(
            StreamEvent(type=EventType.TOOL_START, tool_name="read_file", args={"path": "src/cli/render.py"})
        )
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="read_file", output="content"))
        renderer.render(StreamEvent(type=EventType.FINAL, text="Done"))

        output = console.export_text()
        self.assertLess(output.index("Ran list_files path=src done"), output.index("Ran read_file path=src/cli/render.py done"))

    def test_error_tools_show_error_status_in_final_summary(self) -> None:
        console, renderer = self._make_renderer()

        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="read_file", args={"path": "missing.py"}))
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="read_file", error="[error] missing"))
        renderer.render(StreamEvent(type=EventType.FINAL, text="Done"))

        output = console.export_text()
        self.assertIn("Ran read_file path=missing.py error", output)
        self.assertIn("└ [error] missing", output)

    def test_no_tool_run_skips_condensed_tool_summary(self) -> None:
        console, renderer = self._make_renderer()

        renderer.render(StreamEvent(type=EventType.FINAL, text="Done"))

        output = console.export_text()
        self.assertNotIn("read_file", output)
        self.assertNotIn("tool\n", output)

    def test_expanded_mode_keeps_tool_names_out_of_spinner_status(self) -> None:
        _, renderer = self._make_renderer(collapsed_tools=False)

        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="read_file"))
        self.assertEqual(renderer.get_status_text().plain, "☄ running tool...")

        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="read_file", output="content"))
        self.assertEqual(renderer.get_status_text().plain, "☄ mapping starlines...")

    def test_live_tool_renderable_includes_args_and_preview(self) -> None:
        _, renderer = self._make_renderer()

        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="search_text", args={"query": "spinner"}))
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="search_text", output="src/cli/ui.py:1"))

        live_output = Console(record=True, width=100)
        live_output.print(renderer.build_live_tool_renderable(now=0.0))
        text = live_output.export_text()
        self.assertIn("Ran search_text query=spinner done", text)
        self.assertIn("└ src/cli/ui.py:1", text)
        self.assertIn("────", text)

    def test_live_tool_renderable_omits_transition_while_tool_is_running(self) -> None:
        _, renderer = self._make_renderer()

        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="read_file", args={"path": "src/cli/ui.py"}))

        live_output = Console(record=True, width=100)
        live_output.print(renderer.build_live_tool_renderable(now=0.0))
        text = live_output.export_text()
        self.assertIn("Ran read_file path=src/cli/ui.py running", text)
        self.assertNotIn("────", text)

    def test_live_tool_renderable_shows_latest_completed_tool_only(self) -> None:
        _, renderer = self._make_renderer()

        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="find_files", args={"pattern": "events.py"}))
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="find_files", output="src/schemas/events.py"))
        renderer.render(StreamEvent(type=EventType.TOOL_START, tool_name="read_file", args={"path": "src/schemas/events.py"}))
        renderer.render(StreamEvent(type=EventType.TOOL_END, tool_name="read_file", output="from __future__ import annotations"))

        live_output = Console(record=True, width=100)
        live_output.print(renderer.build_live_tool_renderable(now=0.0))
        text = live_output.export_text()
        self.assertIn("Ran read_file path=src/schemas/events.py done", text)
        self.assertIn("────", text)
        self.assertNotIn("Ran find_files pattern=events.py done", text)


if __name__ == "__main__":
    unittest.main()
