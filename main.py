"""Comet entrypoint.

Installed as the `comet` console script via pyproject.toml.
"""

from __future__ import annotations

import typer

from cli.ui import run_shell

app = typer.Typer(
    name="comet",
    help="Comet — agentic CLI coding assistant.",
    no_args_is_help=False,
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Launch the interactive Comet shell when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        run_shell()


if __name__ == "__main__":
    app()
