"""Streaming event rendering for the interactive shell."""

from __future__ import annotations

import json
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from schemas.events import EventType, StreamEvent


class EventRenderer:
    def __init__(
        self,
        console: Console,
        set_status: Callable[[str], None] | None = None,
    ) -> None:
        self.console = console
        self._set_status = set_status
        self._buffer: list[str] = []

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
        elif event.type == EventType.ERROR:
            self.console.print(f"\n  [bold red]error:[/bold red] {event.error}\n")
        elif event.type == EventType.FINAL:
            self._render_final(event)
        self._update_status_for_event(event)

    def _render_token(self, event: StreamEvent) -> None:
        text = event.text or ""
        if not text:
            return
        self._buffer.append(text)

    def _render_tool_start(self, event: StreamEvent) -> None:
        args = event.args or {}
        args_str = json.dumps(args, ensure_ascii=True) if args else "{}"
        self.console.print(
            Text.assemble(
                ("  tool ", "dim"),
                (event.tool_name or "unknown", "bold bright_cyan"),
                (" ", "white"),
                (args_str, "dim white"),
            )
        )

    def _render_tool_end(self, event: StreamEvent) -> None:
        if event.error:
            self.console.print(f"  [red]error:[/red] {event.error}")
            return

        if event.output:
            lines = event.output.splitlines() or [event.output]
            preview_count = 2
            shown = lines[:preview_count]
            remaining = max(len(lines) - preview_count, 0)
            preview = "\n".join(shown).strip()
            if not preview:
                preview = "[no output]"
            if remaining > 0:
                preview = f"{preview}\n… +{remaining} more line(s)"
            self.console.print(
                Panel(
                    preview,
                    border_style="bright_black",
                    padding=(0, 1),
                    expand=True,
                )
            )

    def _render_final(self, event: StreamEvent) -> None:
        text = "".join(self._buffer).strip()
        if not text:
            text = (event.text or "").strip()
        if not text:
            text = "No final response text was produced."

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
        self._buffer.clear()

    def _render_limit(self, event: StreamEvent) -> None:
        lines = [event.reason or "Run hit a limit."]
        if event.partial_findings:
            lines.append("")
            lines.append("Partial findings:")
            lines.extend(f"- {item}" for item in event.partial_findings[:5])
        if event.suggestion:
            lines.append("")
            lines.append(f"Suggestion: {event.suggestion}")
        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold yellow]limit reached[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
                expand=True,
            )
        )

    def _render_attempt_retry(self, event: StreamEvent) -> None:
        reason = event.reason or "Attempt did not converge."
        self.console.print(
            Text.assemble(
                ("  retry ", "dim"),
                ("trying a different approach", "bold bright_magenta"),
                (" — ", "dim"),
                (reason, "dim white"),
            )
        )

    def _update_status_for_event(self, event: StreamEvent) -> None:
        if self._set_status is None:
            return
        if event.type == EventType.TOOL_START:
            tool = event.tool_name or "tool"
            self._set_status(f"[bright_cyan]☄ calibrating {tool}...[/bright_cyan]")
            return
        if event.type == EventType.TOOL_END:
            self._set_status("[bright_cyan]☄ mapping starlines...[/bright_cyan]")
            return
        if event.type == EventType.TOKEN:
            self._set_status("[bright_cyan]☄ composing answer...[/bright_cyan]")
            return
        if event.type == EventType.FINAL:
            self._set_status("[bright_cyan]☄ docking complete[/bright_cyan]")
            return
        if event.type == EventType.LIMIT:
            self._set_status("[yellow]☄ rerouting after limit...[/yellow]")
            return
        if event.type == EventType.ATTEMPT_RETRY:
            self._set_status("[bright_magenta]☄ switching approach...[/bright_magenta]")
