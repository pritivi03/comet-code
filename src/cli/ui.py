"""Basic interactive CLI shell for Comet.

Shows a space-themed banner, then loops on a bordered multi-line input
box (opencode-style) for user tasks. Slash commands let the user inspect
and switch the current mode and model.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
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

    Submitted with esc → enter; plain enter inserts a newline. Ctrl-D / Ctrl-C
    raise EOFError / KeyboardInterrupt so the outer loop can exit cleanly.
    """
    text_area = TextArea(
        multiline=True,
        wrap_lines=True,
        prompt="› ",
        style="class:text-area",
        focus_on_click=True,
        scrollbar=False,
    )

    frame = Frame(text_area, title="comet")

    def get_toolbar() -> HTML:
        return HTML(
            " <label>mode</label> <mode>{mode}</mode>"
            "   <label>model</label> <model>{model}</model>"
            "   <label>esc·enter to submit · /help for commands</label>".format(
                mode=state.mode.value,
                model=state.model.label,
            )
        )

    toolbar_window = Window(
        FormattedTextControl(get_toolbar),
        height=1,
        style="class:toolbar",
    )

    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _submit(event):  # noqa: ANN001
        event.app.exit(result=text_area.text)

    @kb.add("c-d")
    def _eof(event):  # noqa: ANN001
        event.app.exit(exception=EOFError)

    @kb.add("c-c")
    def _interrupt(event):  # noqa: ANN001
        event.app.exit(exception=KeyboardInterrupt)

    app: Application[str] = Application(
        layout=Layout(HSplit([frame, toolbar_window])),
        key_bindings=kb,
        style=_INPUT_STYLE,
        full_screen=False,
        mouse_support=True,
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


# ---------------------------------------------------------------------------
# Shell entrypoint
# ---------------------------------------------------------------------------


def run_shell() -> None:
    """Entrypoint for the interactive Comet shell."""
    console = Console()
    state = ShellState()
    orchestrator = Orchestrator()

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

        try:
            with console.status(
                "[bright_cyan]☄ actualizing star map...[/bright_cyan]",
                spinner="dots12",
            ) as status:
                renderer = EventRenderer(console, set_status=status.update)
                orchestrator.run_task(
                    user_request=text,
                    mode=state.mode,
                    model=state.model,
                    on_event=renderer.render,
                )
        except Exception as exc:
            console.print(f"\n  [bold red]error:[/bold red] {exc}\n")
