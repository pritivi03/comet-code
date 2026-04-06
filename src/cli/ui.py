"""Basic interactive CLI shell for Comet.

For now this is a placeholder UI: it shows a space-themed banner with a
comet graphic and gives the user a multi-line textbox to type into. It
does not yet wire into the agent loop.
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
    # Title styled with a cyan→magenta gradient feel.
    title = Text("ClaudeCode", style="bold bright_cyan")
    subtitle = Text(_TAGLINE, style="dim white")

    # Color the comet art line-by-line for a gradient look.
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


def _build_prompt_session() -> PromptSession:
    """Build the prompt_toolkit session used for the textbox."""
    bindings = KeyBindings()

    @bindings.add("c-d")
    def _(event):  # noqa: ANN001 — prompt_toolkit signature
        """Ctrl-D exits the shell."""
        event.app.exit(exception=EOFError)

    style = Style.from_dict(
        {
            "prompt": "ansibrightcyan bold",
            "continuation": "ansibrightblue",
        }
    )

    return PromptSession(
        message=HTML("<prompt>› </prompt>"),
        multiline=True,
        prompt_continuation=lambda width, line_number, is_soft_wrap: HTML(
            "<continuation>· </continuation>"
        ),
        key_bindings=bindings,
        style=style,
    )


def run_shell() -> None:
    """Entrypoint for the interactive Comet shell."""
    console = Console()
    console.print()
    console.print(_build_banner())
    console.print(
        "[dim]press [bold]esc → enter[/bold] to submit, [bold]ctrl-d[/bold] to quit[/dim]\n"
    )

    session = _build_prompt_session()

    while True:
        try:
            text = session.prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bright_magenta]✦ goodbye[/bright_magenta]\n")
            return

        text = text.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            console.print("\n[bright_magenta]✦ goodbye[/bright_magenta]\n")
            return

        # Placeholder echo — the agent loop is not wired up yet.
        console.print(
            Panel(
                Text(text, style="white"),
                title="[bright_cyan]received[/bright_cyan]",
                border_style="cyan",
            )
        )
