"""Prompt Toolkit completer for Comet slash commands.

Provides a simple WordCompleter that suggests all commands defined in
`cli.commands.SLASH_COMMANDS`.  The completer is used by the interactive UI
so that pressing <Tab> (or any configured completion key) shows available
commands.
"""

from prompt_toolkit.completion import WordCompleter

from .commands import SLASH_COMMANDS

# Case‑insensitive, matches whole words (including the leading '/')
command_completer = WordCompleter(
    SLASH_COMMANDS,
    ignore_case=True,
    sentence=True,
)
