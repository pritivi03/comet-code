"""Basic interactive CLI shell for Comet.

Shows a space-themed banner, then loops on a bordered multi-line input
box (opencode-style) for user tasks. Slash commands let the user inspect
and switch the current mode and model.
"""

from __future__ import annotations

import difflib

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
import itertools
from rich.text import Text

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from cli.commands import handle_command
from cli.completer import command_completer
from cli.render import EventRenderer
from cli.state import ShellState
from core.orchestrator import Orchestrator


# A small comet ASCII drawing. The trailing dots/dashes form the tail and
# the asterisk is the comet head. Rendered with a cyan -> white gradient.
_COMET_LINES = [
    "         . ·  ✦         .       ✦      ·",
    "    ·          .      ·                 .",
    "                                  ✦",
    "       ·   . ·  ☄ ═══════════════╗",
    "                  ╲╲╲╲╲╲╲╲        ╚═══✦",
    "                   ╲╲╲╲╲╲          ·",
    "    ✦       ·       ╲╲╲╲       .",
    "                     ╲╲                ✦",
    "         ·      ✦         .       ·",
]

_TAGLINE = "agentic coding assistant — type a task below"


def _build_banner() -> Panel:
    """Render the space banner with comet art and the ClaudeCode title."""
    title = Text("CometCode", style="bold bright_cyan")
    subtitle = Text(_TAGLINE, style="dim white")

    art = Text()
    palette = [
        "bright_white",
        "bright_cyan",
        "cyan",
        "bright_blue",
        "blue",
        "magenta",
        "bright_magenta",
        "cyan",
        "bright_cyan",
    ]
    for line, color in zip(_COMET_LINES, palette):
        art.append(line + "\n", style=color)

    body = Group(
        Align.center(art),
        Align.center(title),
        Align.center(subtitle),
    )

    return Panel(
        body,
        title="[bold magenta]✦ comet ✦[/bold magenta]",
        border_style="bright_blue",
        padding=(1, 4),
    )

_INPUT_STYLE = Style.from_dict(
    {
        # Frame border + title
        "frame.border": "ansibrightblue",
        "frame.label": "ansibrightcyan bold",
        # Text inside the box
        "text-area": "ansiwhite",
        "text-area.prompt": "ansibrightcyan bold",
        # Bottom status line
        "toolbar": "bg:#1a1a2e #c0c0ff",
        "toolbar.label": "bg:#1a1a2e #8888aa",
        "toolbar.mode": "bg:#1a1a2e ansibrightmagenta bold",
        "toolbar.model": "bg:#1a1a2e ansibrightcyan bold",
    }
)


def _read_boxed_input(state: ShellState) -> str:
    """Display a bordered text box and return the user's submitted text.

    Submitted with enter; Ctrl-J inserts a newline. Ctrl-D / Ctrl-C
    raise EOFError / KeyboardInterrupt so the outer loop can exit cleanly.
    """
    text_area = TextArea(
        multiline=True,
        wrap_lines=True,
        prompt="› ",
        style="class:text-area",
        focus_on_click=True,
        scrollbar=False,
        completer=command_completer,
        complete_while_typing=True,
    )

    frame = Frame(text_area, title="comet")

    def get_toolbar() -> HTML:
        tools_mode = "collapsed" if state.tool_view_collapsed else "expanded"
        return HTML(
            " <label>mode</label> <mode>{mode}</mode>"
            "   <label>model</label> <model>{model}</model>"
            "   <label>tools</label> <mode>{tools}</mode>"
            "   <label>enter submit · ctrl-j newline · ctrl-o toggle tools · /help</label>".format(
                mode=state.mode.value,
                model=state.model.label,
                tools=tools_mode,
            )
        )

    toolbar_window = Window(
        FormattedTextControl(get_toolbar),
        height=1,
        style="class:toolbar",
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):  # noqa: ANN001
        event.app.exit(result=text_area.text)

    @kb.add("c-j")
    def _newline(event):  # noqa: ANN001
        event.current_buffer.insert_text("\n")

    @kb.add("c-d")
    def _eof(event):  # noqa: ANN001
        event.app.exit(exception=EOFError)

    @kb.add("c-c")
    def _interrupt(event):  # noqa: ANN001
        event.app.exit(exception=KeyboardInterrupt)

    @kb.add("c-o")
    def _toggle_tools_view(event):  # noqa: ANN001
        state.tool_view_collapsed = not state.tool_view_collapsed
        event.app.invalidate()

    app: Application[str] = Application(
        layout=Layout(HSplit([frame, toolbar_window])),
        key_bindings=kb,
        style=_INPUT_STYLE,
        full_screen=False,
        mouse_support=False,
        # Erase the rendered box on exit so the next iteration redraws in
        # the same spot — gives the illusion of a fixed bottom-pinned UI
        # instead of leaving stale toolbars in the scrollback.
        erase_when_done=True,
    )

    return app.run()


# ---------------------------------------------------------------------------
# User-message echo
# ---------------------------------------------------------------------------

# Echoed user messages get truncated past these limits so the scrollback
# doesn't fill up with giant pasted blocks.
_ECHO_MAX_LINES = 4
_ECHO_MAX_CHARS = 240


def _truncate_for_echo(text: str) -> str:
    """Shorten text for the user-message echo panel."""
    lines = text.splitlines()
    truncated_lines = False
    if len(lines) > _ECHO_MAX_LINES:
        lines = lines[:_ECHO_MAX_LINES]
        truncated_lines = True

    shortened = "\n".join(lines)
    if len(shortened) > _ECHO_MAX_CHARS:
        shortened = shortened[:_ECHO_MAX_CHARS].rstrip()
        truncated_lines = True

    if truncated_lines:
        shortened += " …"
    return shortened


def _format_user_echo(text: str) -> Panel:
    return Panel(
        Text(_truncate_for_echo(text), style="white"),
        border_style="bright_black",
        padding=(0, 1),
        expand=True,
    )


def _format_tool_args(args: dict[str, object]) -> str:
    if not args:
        return "{}"
    parts: list[str] = []
    for key, value in args.items():
        rendered = repr(value) if isinstance(value, str) else str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _preview_text_block(text: str, *, max_lines: int = 5, max_chars: int = 240) -> str:
    lines = text.splitlines() or [text]
    clipped = lines[:max_lines]
    preview = "\n".join(clipped)
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3].rstrip() + "..."
    elif len(lines) > max_lines:
        preview += "\n..."
    return preview


def _build_diff_renderable(old_text: str, new_text: str, max_lines: int = 30) -> Text:
    """Return a Rich Text with colored unified-diff lines (red=removed, green=added)."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=3))
    result = Text()
    for i, line in enumerate(diff):
        if i >= max_lines:
            result.append("...\n", style="dim")
            break
        if line.startswith("---") or line.startswith("+++"):
            result.append(line + "\n", style="dim")
        elif line.startswith("@@"):
            result.append(line + "\n", style="bold cyan")
        elif line.startswith("-"):
            result.append(line + "\n", style="bold red")
        elif line.startswith("+"):
            result.append(line + "\n", style="bold green")
        else:
            result.append(line + "\n", style="default")
    if not diff:
        result.append("(no changes)", style="dim")
    return result


def _format_proposal_lines(proposal: dict[str, object]) -> list[str | Text]:
    tool_name = str(proposal.get("tool_name", "tool"))
    args = proposal.get("args", {})
    reason = str(proposal.get("reason") or "").strip()
    safe_args = args if isinstance(args, dict) else {}

    lines: list[str | Text] = []
    if tool_name == "replace_text":
        path = str(safe_args.get("path", ""))
        replace_all_flag = bool(safe_args.get("replace_all", False))
        old_text = str(safe_args.get("old_text", ""))
        new_text = str(safe_args.get("new_text", ""))
        lines.append(f"replace_text {path}".rstrip())
        lines.append(f"mode: {'replace all matches' if replace_all_flag else 'replace first exact match'}")
        if reason:
            lines.append(f"why: {reason}")
        lines.append(_build_diff_renderable(old_text, new_text))
        return lines

    if tool_name == "write_file":
        path = str(safe_args.get("path", ""))
        content = str(safe_args.get("content", ""))
        create_dirs = bool(safe_args.get("create_dirs", False))
        lines.append(f"write_file {path}".rstrip())
        lines.append(f"create_dirs: {'true' if create_dirs else 'false'}")
        if reason:
            lines.append(f"why: {reason}")
        lines.append("content:")
        lines.append(_preview_text_block(content))
        return lines

    lines.append(f"{tool_name} {_format_tool_args(safe_args)}".rstrip())
    if reason:
        lines.append(f"why: {reason}")
    return lines  # type: ignore[return-value]


def _prompt_tool_approval(
    console: Console,
    live: Live,
    renderer: EventRenderer,
    proposals: list[dict[str, object]],
) -> bool:
    live.stop()
    try:
        renderer.persist_tool_history_snapshot()

        renderables: list[object] = []
        for idx, proposal in enumerate(proposals, start=1):
            if idx > 1:
                renderables.append(Text(""))
            renderables.append(Text(f"{idx}.", style="bold"))
            for item in _format_proposal_lines(proposal):
                if isinstance(item, Text):
                    renderables.append(Text.assemble(("   ", ""), item))
                else:
                    renderables.append(Text(f"   {item}"))

        console.print(
            Panel(
                Group(*renderables),
                title="[bold bright_cyan]proposed change[/bold bright_cyan]",
                border_style="bright_blue",
                padding=(0, 1),
                expand=True,
            )
        )

        answer = console.input("[bold bright_cyan]apply this change?[/bold bright_cyan] [dim](y/N)[/dim] ").strip().lower()
        approved = answer in {"y", "yes"}
        if approved:
            console.print("[bright_cyan]approved[/bright_cyan]")
        else:
            console.print("[yellow]rejected[/yellow]")
        return approved
    finally:
        live.start(refresh=True)


# ---------------------------------------------------------------------------
# Shell entrypoint
# ---------------------------------------------------------------------------


def run_shell(api_key: str = "") -> None:
    """Entrypoint for the interactive Comet shell."""
    console = Console()
    state = ShellState()
    orchestrator = Orchestrator(api_key=api_key)

    console.print()
    console.print(_build_banner())
    console.print()

    while True:
        try:
            text = _read_boxed_input(state)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bright_magenta]✦ goodbye[/bright_magenta]\n")
            return

        text = text.strip()
        if not text:
            continue

        result = handle_command(text, console, state, orchestrator)
        if result.should_exit:
            console.print("\n[bright_magenta]✦ goodbye[/bright_magenta]\n")
            return
        if result.handled:
            continue

        # Echo the user's message in a small panel above the input box
        console.print(_format_user_echo(text))

        renderer = EventRenderer(
            console,
            collapsed_tools=state.tool_view_collapsed,
        )
        spinner = Spinner(
            "dots12",
            text=renderer.get_status_text(),
            style="bright_cyan",
        )
        # Shimmer effect: a single bright highlight moves left‑to‑right across the status text
        _shimmer_counter = itertools.count()

        def _build_run_live_display() -> Group:
            status_txt = renderer.get_status_text()
            # Ensure we have a plain string length to work with; fall back to 0 length safety
            # Determine length of the loading text (exclude token count and elapsed time)
            loading_len = len(renderer._status_text.plain) if getattr(renderer, "_status_text", None) else 0
            if loading_len:
                idx = next(_shimmer_counter) % loading_len
                # Apply a bright magenta highlight to the current character position (only on loading text)
                status_txt.stylize("bright_magenta", idx, idx + 1)
            spinner.update(text=status_txt)
            items: list[object] = [
                spinner
            ]
            if renderer.should_render_live_tool_row():
                items.append(renderer.build_live_tool_renderable())
            return Group(*items)

        try:
            with Live(
                console=console,
                transient=True,
                refresh_per_second=12,
                get_renderable=_build_run_live_display,
            ) as live:
                orchestrator.run_task(
                    user_request=text,
                    mode=state.mode,
                    model=state.model,
                    on_event=renderer.render,
                    request_approval=lambda proposals: _prompt_tool_approval(console, live, renderer, proposals),
                )
        except Exception as exc:
            console.print(f"\n  [bold red]error:[/bold red] {exc}\n")
        finally:
            # Flush the response panel after Live exits so it lands in the
            # terminal's normal scroll buffer — no cursor-up magic from the
            # Live context can confuse the terminal's scrollback.
            renderer.flush_final()
            state.last_tool_history = renderer.get_tool_history()
            console.print(Text(f"Cooked for {renderer.get_elapsed_str()}", style="dim"))
