"""Streaming event rendering for the interactive shell."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from rich.console import Console
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from schemas.events import EventType, StreamEvent

_TOOL_DISPLAY: dict[str, tuple[str, str | None]] = {
    "list_files":   ("List",    "path"),
    "search_text":  ("Search",  "pattern"),
    "find_files":   ("Find",    "pattern"),
    "print_tree":   ("Tree",    "path"),
    "read_file":    ("Read",    "path"),
    "read_range":   ("Read",    "path"),
    "write_file":   ("Write",   "path"),
    "replace_text": ("Edit",    "path"),
}


@dataclass
class ToolHistoryEntry:
    tool_name: str
    args_json: str
    reason: str | None
    status: str = "running"
    preview: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "args_json": self.args_json,
            "reason": self.reason,
            "status": self.status,
            "preview": self.preview,
            "error": self.error,
        }


class EventRenderer:
    def __init__(
        self,
        console: Console,
        collapsed_tools: bool = True,
    ) -> None:
        self.console = console
        self._buffer: list[str] = []
        self._collapsed_tools = collapsed_tools
        self._tool_history: list[ToolHistoryEntry] = []
        self._active_tool_idx: int | None = None
        self._status_text = Text("☄ actualizing star map...", style="bright_cyan")
        self._show_post_tool_transition = False
        self._persisted_tool_history_count = 0
        self._token_count: int = 0
        self._start_time: float = time.monotonic()
        self._pending_final_text: str | None = None

    def get_tool_history(self) -> list[dict]:
        return [item.to_dict() for item in self._tool_history]

    def get_elapsed_str(self) -> str:
        secs = int(time.monotonic() - self._start_time)
        if secs >= 60:
            return f"{secs // 60}m {secs % 60:02d}s"
        return f"{secs}s"

    def get_status_text(self) -> Text:
        txt = self._status_text.copy()
        elapsed = self.get_elapsed_str()
        # Separator before token count / elapsed time for clarity
        if self._token_count > 0:
            txt.append(" |", style="dim")
            n = self._token_count
            label = f"~{n/1000:.1f}k" if n >= 1000 else f"~{n}"
            txt.append(f" {label} tokens", style="dim")
            txt.append(" |", style="dim")
        else:
            txt.append(" |", style="dim")
        txt.append(f" {elapsed}", style="dim")
        return txt

    def persist_tool_history_snapshot(self) -> None:
        if not self._collapsed_tools:
            return
        if self._persisted_tool_history_count >= len(self._tool_history):
            return
        self.console.print()
        self.console.print(self._build_collapsed_tool_summary(start_idx=self._persisted_tool_history_count))
        self._persisted_tool_history_count = len(self._tool_history)

    def should_render_live_tool_row(self) -> bool:
        return self._collapsed_tools and bool(self._tool_history)

    def build_live_tool_renderable(self, now: float | None = None) -> Group | Text:
        if not self.should_render_live_tool_row():
            return Text("")

        visible_idx = self._visible_tool_index()
        if visible_idx is None:
            return Text("")

        entry = self._tool_history[visible_idx]
        human_name, key_arg = self._format_tool_display(entry)
        lines: list[Text] = [
            Text("  tool", style="dim"),
            Text.assemble(
                ("    ", "default"),
                (f"{self._status_dot(entry.status)} ", self._status_style(entry.status)),
                (human_name, "bold bright_cyan"),
                (" ", "default"),
                (key_arg, "white"),
                (" ", "default"),
                (entry.status, "dim"),
            ),
        ]
        preview = self._tool_secondary_line(entry)
        if preview is not None:
            lines.append(preview)
        if self._show_post_tool_transition:
            lines.append(
                Text.assemble(
                    ("    ", "default"),
                    ("────", "dim"),
                )
            )
        return Group(*lines)

    def render(self, event: StreamEvent) -> None:
        if event.type == EventType.TOKEN:
            self._render_token(event)
        elif event.type == EventType.TOOL_START:
            self._render_tool_start(event)
        elif event.type == EventType.TOOL_END:
            self._render_tool_end(event)
        elif event.type == EventType.LIMIT:
            self._render_limit(event)
        elif event.type == EventType.ATTEMPT_RETRY:
            self._render_attempt_retry(event)
        elif event.type == EventType.USAGE:
            self._render_usage(event)
        elif event.type == EventType.ERROR:
            self.console.print(f"\n  [bold red]error:[/bold red] {event.error}\n")
        elif event.type == EventType.FINAL:
            self._render_final(event)
        self._update_status_for_event(event)

    def _render_token(self, event: StreamEvent) -> None:
        text = event.text or ""
        if text:
            self._buffer.append(text)
            self._token_count += max(1, len(text) // 4)
        if self._tool_history:
            self._show_post_tool_transition = True

    def _render_tool_start(self, event: StreamEvent) -> None:
        args = event.args or {}
        args_str = json.dumps(args, ensure_ascii=True) if args else "{}"
        reason = (event.reason or "").strip() or None
        if reason and len(reason) > 160:
            reason = reason[:157].rstrip() + "..."

        entry = ToolHistoryEntry(
            tool_name=event.tool_name or "unknown",
            args_json=args_str,
            reason=reason,
            status="running",
        )
        self._tool_history.append(entry)
        self._active_tool_idx = len(self._tool_history) - 1
        self._show_post_tool_transition = False

        if self._collapsed_tools:
            return

        human_name, key_arg = self._format_tool_display(entry)
        self.console.print(
            Text.assemble(
                ("  tool ", "dim"),
                (human_name, "bold bright_cyan"),
                (" ", "default"),
                (key_arg, "white"),
            )
        )
        if entry.reason:
            self.console.print(Text.assemble(("    why ", "dim"), (entry.reason, "italic dim")))

    def _render_tool_end(self, event: StreamEvent) -> None:
        entry: ToolHistoryEntry | None = None
        if self._active_tool_idx is not None and 0 <= self._active_tool_idx < len(self._tool_history):
            entry = self._tool_history[self._active_tool_idx]

        if entry is None:
            args = event.args or {}
            args_str = json.dumps(args, ensure_ascii=True) if args else "{}"
            entry = ToolHistoryEntry(
                tool_name=event.tool_name or "unknown",
                args_json=args_str,
                reason=None,
                status="done",
            )
            self._tool_history.append(entry)

        if event.error:
            entry.status = "error"
            entry.error = event.error
            self._active_tool_idx = None
            self._show_post_tool_transition = True
            if not self._collapsed_tools:
                self.console.print(f"  [red]error:[/red] {event.error}")
            return

        if event.output:
            self._token_count += max(1, len(event.output) // 4)
            lines = event.output.splitlines() or [event.output]
            shown = lines[:2]
            remaining = max(len(lines) - 2, 0)
            preview = "\n".join(shown).strip() or "[no output]"
            if remaining > 0:
                preview = f"{preview}\n… +{remaining} more line(s)"
            entry.preview = preview
            entry.status = "done"

            if not self._collapsed_tools:
                self.console.print(
                    Panel(
                        preview,
                        border_style="bright_black",
                        padding=(0, 1),
                        expand=True,
                )
                )
        else:
            entry.status = "done"
        self._active_tool_idx = None
        self._show_post_tool_transition = True

    def _render_final(self, event: StreamEvent) -> None:
        text = "".join(self._buffer).strip()
        if not text:
            text = (event.text or "").strip()
        if not text:
            text = "No final response text was produced."
        self._pending_final_text = text
        self._buffer.clear()
        self._show_post_tool_transition = False

    def flush_final(self) -> None:
        """Print the final response panel outside any Live context.

        Called after the Live spinner exits so the terminal's normal scroll
        buffer is clean — no cursor-up manipulation can confuse scrollback.
        """
        text = self._pending_final_text
        if text is None:
            return
        self._pending_final_text = None

        if self._collapsed_tools and self._persisted_tool_history_count < len(self._tool_history):
            self.console.print()
            self.console.print(self._build_collapsed_tool_summary(start_idx=self._persisted_tool_history_count))
            self._persisted_tool_history_count = len(self._tool_history)

        self.console.print()
        self.console.print(
            Panel(
                Markdown(text),
                title="[bold bright_cyan]response[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2),
                expand=True,
            )
        )
        self.console.print()

    def _render_limit(self, event: StreamEvent) -> None:
        self._show_post_tool_transition = bool(self._tool_history)
        return

    def _render_attempt_retry(self, event: StreamEvent) -> None:
        self._show_post_tool_transition = bool(self._tool_history)
        reason = event.reason or "Attempt did not converge."
        self.console.print(
            Text.assemble(
                ("  retry ", "dim"),
                ("trying a different approach", "bold bright_magenta"),
                (" — ", "dim"),
                (reason, "dim white"),
            )
        )

    def _render_usage(self, event: StreamEvent) -> None:
        total = event.total_tokens or 0
        if total <= 0:
            return
        label = f"↓ ~{total:,} tokens"
        if event.estimated:
            label += " (est.)"
        self.console.print(Text(label, style="dim"))

    def _visible_tool_index(self) -> int | None:
        if not self._tool_history:
            return None
        if self._active_tool_idx is not None and 0 <= self._active_tool_idx < len(self._tool_history):
            return self._active_tool_idx
        return len(self._tool_history) - 1

    def _build_collapsed_tool_summary(self, start_idx: int = 0) -> Group:
        lines = [Text("  tool", style="dim")]
        for entry in self._tool_history[start_idx:]:
            human_name, key_arg = self._format_tool_display(entry)
            lines.append(
                Text.assemble(
                    ("    ", "default"),
                    (f"{self._status_dot(entry.status)} ", self._status_style(entry.status)),
                    (human_name, "bold bright_cyan"),
                    (" ", "default"),
                    (key_arg, "white"),
                    (" ", "default"),
                    (entry.status, "dim"),
                )
            )
            preview = self._tool_secondary_line(entry)
            if preview is not None:
                lines.append(preview)
        return Group(*lines)

    def _format_tool_display(self, entry: ToolHistoryEntry) -> tuple[str, str]:
        """Return (human_label, key_arg_value) for clean title rendering."""
        human_label, primary_key = _TOOL_DISPLAY.get(entry.tool_name, (entry.tool_name, None))
        args = self._load_args(entry.args_json)
        key_value = ""
        if primary_key and primary_key in args:
            raw = self._format_arg(primary_key, args[primary_key])
            key_value = raw.split("=", 1)[1].strip('"') if "=" in raw else raw
        elif args:
            first_key, first_val = next(iter(args.items()))
            raw = self._format_arg(first_key, first_val)
            key_value = raw.split("=", 1)[1].strip('"') if "=" in raw else raw
        return human_label, key_value

    def _format_tool_invocation(self, entry: ToolHistoryEntry) -> str:
        args = self._load_args(entry.args_json)
        if not args:
            return entry.tool_name

        parts = [entry.tool_name]
        for key, value in args.items():
            parts.append(self._format_arg(key, value))
        return " ".join(parts)

    def _tool_secondary_line(self, entry: ToolHistoryEntry) -> Text | None:
        detail: str | None = None
        style = "dim"
        if entry.error:
            detail = entry.error
            style = "red"
        elif entry.preview:
            detail = entry.preview.splitlines()[0].strip()
        elif entry.reason:
            detail = entry.reason

        if not detail:
            return None

        if len(detail) > 120:
            detail = detail[:117].rstrip() + "..."
        return Text.assemble(
            ("    ", "default"),
            ("└ ", "dim"),
            (detail, style),
        )

    def _load_args(self, args_json: str) -> dict[str, object]:
        try:
            loaded = json.loads(args_json)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _format_arg(self, key: str, value: object) -> str:
        if isinstance(value, str):
            rendered = value if " " not in value else json.dumps(value, ensure_ascii=True)
            return f"{key}={rendered}"
        if isinstance(value, bool):
            return f"{key}={'true' if value else 'false'}"
        if isinstance(value, (int, float)):
            return f"{key}={value}"
        return f"{key}={json.dumps(value, ensure_ascii=True)}"

    def _status_dot(self, status: str) -> str:
        if status == "error":
            return "●"
        if status == "running":
            return "●"
        return "●"

    def _status_style(self, status: str) -> str:
        if status == "error":
            return "red"
        if status == "running":
            return "yellow"
        return "green"

    def _update_status_for_event(self, event: StreamEvent) -> None:
        if event.type == EventType.TOOL_START:
            self._status_text = Text("☄ running tool...", style="bright_cyan")
            return
        if event.type == EventType.TOOL_END:
            self._status_text = Text("☄ mapping starlines...", style="bright_cyan")
            return
        if event.type == EventType.TOKEN:
            self._status_text = Text("☄ composing answer...", style="bright_cyan")
            return
        if event.type == EventType.FINAL:
            self._status_text = Text("☄ docking complete", style="bright_cyan")
            return
        if event.type == EventType.LIMIT:
            self._status_text = Text("☄ wrapping up an answer...", style="bright_cyan")
            return
        if event.type == EventType.ATTEMPT_RETRY:
            self._status_text = Text("☄ switching approach...", style="bright_magenta")
            return
        if event.type == EventType.ERROR:
            self._status_text = Text("☄ run failed", style="red")
