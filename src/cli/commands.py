"""Slash-command handling for the interactive shell.

Commands return a `CommandResult` so the shell loop knows whether to keep
running, exit, or just continue after printing feedback. Anything that
doesn't start with `/` is treated as a user task and handled by the shell.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.state import ShellState
from core.orchestrator import Orchestrator
from llm.models import AVAILABLE_MODELS, find_model
from schemas.task import TaskMode


@dataclass
class CommandResult:
    handled: bool          # True if input was a recognized slash command
    should_exit: bool = False


def _print_modes(console: Console, state: ShellState) -> None:
    table = Table(title="modes", title_style="bold bright_cyan", border_style="cyan")
    table.add_column("name", style="bright_white")
    table.add_column("current", style="bright_magenta")
    for m in TaskMode:
        marker = "●" if m == state.mode else ""
        table.add_row(m.value, marker)
    console.print(table)


def _print_models(console: Console, state: ShellState) -> None:
    table = Table(title="models", title_style="bold bright_cyan", border_style="cyan")
    table.add_column("alias", style="bright_white")
    table.add_column("slug", style="dim")
    table.add_column("label", style="white")
    table.add_column("current", style="bright_magenta")
    for m in AVAILABLE_MODELS:
        marker = "●" if m.slug == state.model.slug else ""
        table.add_row(m.short, m.slug, m.label, marker)
    console.print(table)


def _print_help(console: Console) -> None:
    table = Table(title="commands", title_style="bold bright_cyan", border_style="cyan")
    table.add_column("command", style="bright_white")
    table.add_column("description", style="white")
    rows = [
        ("/help",            "show this help"),
        ("/mode",            "show current mode and list available modes"),
        ("/mode <name>",     "switch to a mode (explain, debug, refactor, implement, plan)"),
        ("/<mode>",          "shortcut: e.g. /plan, /debug, /explain, /refactor, /implement"),
        ("/model",           "show current model and list available models"),
        ("/model <name>",    "switch model (by alias, slug, or label)"),
        ("/tools",           "show last run tool-call history"),
        ("/tools <toggle|collapse|expand>", "set live tool view mode"),
        ("/clear",           "clear conversation history"),
        ("/exit, /quit",     "leave the shell"),
    ]
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(table)


def _cmd_mode(console: Console, state: ShellState, args: list[str]) -> None:
    if not args:
        _print_modes(console, state)
        return
    name = args[0].lower()
    try:
        state.mode = TaskMode(name)
    except ValueError:
        console.print(
            f"[red]unknown mode:[/red] {name}  "
            f"[dim](try: {', '.join(TaskMode.names())})[/dim]"
        )
        return
    console.print(f"[bright_cyan]mode →[/bright_cyan] [bold]{state.mode.value}[/bold]")


def _cmd_model(console: Console, state: ShellState, args: list[str]) -> None:
    if not args:
        _print_models(console, state)
        return
    query = " ".join(args)
    found = find_model(query)
    if found is None:
        console.print(f"[red]unknown model:[/red] {query}  [dim](try /model for the list)[/dim]")
        return
    state.model = found
    console.print(
        f"[bright_cyan]model →[/bright_cyan] [bold]{found.label}[/bold] "
        f"[dim]({found.slug})[/dim]"
    )


def _print_tool_history(console: Console, state: ShellState) -> None:
    mode_label = "collapsed" if state.tool_view_collapsed else "expanded"
    if not state.last_tool_history:
        console.print(
            f"[bright_cyan]tools view →[/bright_cyan] [bold]{mode_label}[/bold]  "
            f"[dim](no tool history yet)[/dim]"
        )
        return

    lines: list[str] = []
    for idx, item in enumerate(state.last_tool_history, start=1):
        status = item.get("status", "unknown")
        tool_name = item.get("tool_name", "tool")
        args_str = item.get("args_json", "{}")
        lines.append(f"{idx}. {tool_name} {args_str} [{status}]")
        reason = item.get("reason")
        if reason:
            lines.append(f"   why: {reason}")
        preview = item.get("preview")
        if preview:
            lines.append(f"   {preview}")
        error = item.get("error")
        if error:
            lines.append(f"   error: {error}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold bright_cyan]tool history ({mode_label})[/bold bright_cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=True,
        )
    )


def handle_command(
    text: str,
    console: Console,
    state: ShellState,
    orchestrator: Orchestrator,
) -> CommandResult:
    """Dispatch a slash-prefixed line. Non-slash lines return handled=False."""
    if not text.startswith("/"):
        return CommandResult(handled=False)

    parts = text[1:].split()
    if not parts:
        return CommandResult(handled=True)

    name, args = parts[0].lower(), parts[1:]

    if name in {"exit", "quit"}:
        return CommandResult(handled=True, should_exit=True)

    if name == "help":
        _print_help(console)
        return CommandResult(handled=True)

    if name == "mode":
        _cmd_mode(console, state, args)
        return CommandResult(handled=True)

    if name == "model":
        _cmd_model(console, state, args)
        return CommandResult(handled=True)

    if name == "clear":
        orchestrator.reset_history()
        state.last_tool_history = []
        console.print("[bright_cyan]history cleared[/bright_cyan]")
        return CommandResult(handled=True)

    if name == "tools":
        if args:
            action = args[0].lower()
            if action == "toggle":
                state.tool_view_collapsed = not state.tool_view_collapsed
            elif action == "collapse":
                state.tool_view_collapsed = True
            elif action == "expand":
                state.tool_view_collapsed = False
            else:
                console.print("[red]unknown /tools option:[/red] " + action)
                return CommandResult(handled=True)

        _print_tool_history(console, state)
        return CommandResult(handled=True)

    # Mode shortcuts: /plan, /debug, /explain, /refactor, /implement
    if name in TaskMode.names():
        _cmd_mode(console, state, [name])
        return CommandResult(handled=True)

    console.print(
        f"[red]unknown command:[/red] /{name}  [dim](try /help)[/dim]"
    )
    return CommandResult(handled=True)
