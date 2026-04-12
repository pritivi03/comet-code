"""Comet entrypoint.

Installed as the `comet` console script via pyproject.toml.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", ".*Pydantic V1.*")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.ui import run_shell
from config import config_file_path, is_valid_key, mask_key, resolve_api_key, save_key

app = typer.Typer(
    name="comet",
    help="Comet — agentic CLI coding assistant.",
    no_args_is_help=False,
    add_completion=False,
)


def _prompt_for_key(console: Console) -> str | None:
    """Ask the user for an OpenRouter API key on first run.

    Returns the key if provided (and optionally saved), or None if skipped.
    """
    console.print(
        Panel(
            Text.from_markup(
                "No OpenRouter API key found.\n\n"
                "Get one at [bold]https://openrouter.ai/keys[/bold]\n\n"
                "You can also skip this and set it later:\n"
                "  • Set [bold]OPENROUTER_API_KEY[/bold] in your environment\n"
                f"  • Or run [bold]/key set <key>[/bold] inside the shell (saves to {config_file_path()})"
            ),
            title="[bold bright_cyan]API Key Setup[/bold bright_cyan]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    while True:
        raw = console.input(
            "[bold bright_cyan]Enter API key[/bold bright_cyan] [dim](or press Enter to skip)[/dim]: "
        ).strip()

        if not raw:
            console.print("[dim]Skipping — you can set it later with /key set <key>[/dim]")
            return None

        if not is_valid_key(raw):
            console.print("[red]Invalid key:[/red] OpenRouter keys must start with [bold]sk-or-[/bold]")
            continue

        save_prompt = console.input(
            f"[dim]Save to {config_file_path()}?[/dim] [bold bright_cyan][Y/n][/bold bright_cyan] "
        ).strip().lower()

        if save_prompt in {"", "y", "yes"}:
            save_key(raw)
            console.print(
                f"[bright_cyan]Saved[/bright_cyan] [dim]{mask_key(raw)}  ({config_file_path()})[/dim]"
            )
        else:
            console.print("[dim]Key not saved — will only be used for this session[/dim]")

        return raw


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Launch the interactive Comet shell when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return

    console = Console()
    api_key = resolve_api_key()

    if api_key is None:
        api_key = _prompt_for_key(console) or ""

    run_shell(api_key=api_key)


if __name__ == "__main__":
    app()
