"""Basic interactive CLI shell for Comet.

For now this is a placeholder UI: it shows a space-themed banner with a
comet graphic and gives the user a multi-line textbox to type into.
Slash commands let the user inspect/switch the current mode and model.
The agent loop is not yet wired in.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from cli.commands import handle_command
from cli.state import ShellState


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
    title = Text("ClaudeCode", style="bold bright_cyan")
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


def _build_prompt_session(state: ShellState) -> PromptSession:
    """Build the prompt_toolkit session used for the textbox.

    The bottom toolbar reflects the live `ShellState` so mode/model
    changes show up immediately after a slash command.
    """
    bindings = KeyBindings()

    @bindings.add("c-d")
    def _(event):  # noqa: ANN001 — prompt_toolkit signature
        """Ctrl-D exits the shell."""
        event.app.exit(exception=EOFError)

    style = Style.from_dict(
        {
            "prompt": "ansibrightcyan bold",
            "continuation": "ansibrightblue",
            "bottom-toolbar": "bg:#1a1a2e #c0c0ff",
            "bottom-toolbar.mode": "bg:#1a1a2e ansibrightmagenta bold",
            "bottom-toolbar.model": "bg:#1a1a2e ansibrightcyan bold",
            "bottom-toolbar.label": "bg:#1a1a2e #8888aa",
        }
    )

    def bottom_toolbar() -> HTML:
        return HTML(
            " <label>mode</label> <mode>{mode}</mode>"
            "   <label>model</label> <model>{model}</model>"
            "   <label>/help for commands</label>".format(
                mode=state.mode.value,
                model=state.model.label,
            )
        )

    return PromptSession(
        message=HTML("<prompt>› </prompt>"),
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: HTML(
            "<continuation>· </continuation>"
        ),
        key_bindings=bindings,
        style=style,
        bottom_toolbar=bottom_toolbar,
    )


def run_shell() -> None:
    """Entrypoint for the interactive Comet shell."""
    console = Console()
    state = ShellState()

    console.print()
    console.print(_build_banner())
    console.print(
        "[dim]press [bold]esc → enter[/bold] to submit, "
        "[bold]ctrl-d[/bold] to quit, [bold]/help[/bold] for commands[/dim]\n"
    )

    session = _build_prompt_session(state)

    while True:
        try:
            text = session.prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bright_magenta]✦ goodbye[/bright_magenta]\n")
            return

        text = text.strip()
        if not text:
            continue

        result = handle_command(text, console, state)
        if result.should_exit:
            console.print("\n[bright_magenta]✦ goodbye[/bright_magenta]\n")
            return
        if result.handled:
            continue

        # Placeholder echo — the agent loop is not wired up yet.
        console.print(
            Panel(
                Text(text, style="white"),
                title=(
                    f"[bright_cyan]received[/bright_cyan] "
                    f"[dim]· {state.mode.value} · {state.model.label}[/dim]"
                ),
                border_style="cyan",
            )
        )
