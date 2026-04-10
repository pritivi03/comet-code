"""Renders interaction steps to the console.

The orchestrator calls `on_step` after each parsed model response.
This module owns all presentation logic — the orchestrator never
touches the console directly.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from schemas.attempt import InteractionStep, ResponseType


def render_step(console: Console, step: InteractionStep) -> None:
    """Render a single interaction step based on its response type."""
    response = step.model_response

    if response.type == ResponseType.TOOL_CALLS:
        _render_tool_calls(console, step)
    elif response.type == ResponseType.EDITS:
        _render_edits(console, step)
    elif response.type == ResponseType.FINAL:
        _render_final(console, step)


def _render_tool_calls(console: Console, step: InteractionStep) -> None:
    for ta in step.tool_actions:
        status_style = "green" if ta.output else "yellow"
        args_str = " ".join(ta.args) if ta.args else ""

        console.print(
            Text.assemble(
                ("  tool ", "dim"),
                (ta.tool_name, f"bold {status_style}"),
                (f" {args_str}", "dim white"),
            )
        )

        if ta.output:
            # Truncate long tool output for display
            display = ta.output if len(ta.output) <= 500 else ta.output[:500] + "\n..."
            console.print(
                Panel(
                    display,
                    border_style="bright_black",
                    padding=(0, 1),
                    expand=True,
                )
            )

        if ta.error:
            console.print(f"  [red]error:[/red] {ta.error}")


def _render_edits(console: Console, step: InteractionStep) -> None:
    response = step.model_response
    if not response.edits:
        return

    for edit in response.edits:
        console.print(
            Text.assemble(
                ("  edit ", "dim"),
                (edit.file_path, "bold bright_cyan"),
                (f" L{edit.start_line}-{edit.end_line}", "dim"),
            )
        )

        # Show a simple before/after diff
        console.print(
            Syntax(
                edit.original,
                "python",
                theme="monokai",
                line_numbers=False,
                background_color="#1a0000",
            )
        )
        console.print(
            Syntax(
                edit.replacement,
                "python",
                theme="monokai",
                line_numbers=False,
                background_color="#001a00",
            )
        )


def _render_final(console: Console, step: InteractionStep) -> None:
    response = step.model_response

    if response.explanation:
        console.print()
        console.print(
            Panel(
                Markdown(response.explanation),
                title="[bold bright_cyan]explanation[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    if response.summary:
        console.print()
        console.print(f"  [bright_cyan]summary[/bright_cyan]  {response.summary}")
        console.print()