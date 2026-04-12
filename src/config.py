"""User configuration and API key management for CometCode.

Config directory: ~/.comet/
Config file:      ~/.comet/config.json
Schema:           {"openrouter_api_key": "<string>"}

Resolution order for the API key:
  1. OPENROUTER_API_KEY environment variable  (always wins)
  2. openrouter_api_key field in ~/.comet/config.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_CONFIG_DIR  = Path.home() / ".comet"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_ENV_VAR     = "OPENROUTER_API_KEY"


# ── Validation & display ─────────────────────────────────────────────────────

def is_valid_key(key: str) -> bool:
    """Return True if key is non-empty and starts with 'sk-or-'."""
    return bool(key) and key.startswith("sk-or-")


def mask_key(key: str) -> str:
    """Return a display-safe version: first 8 chars + '...' + last 4 chars.

    Example: 'sk-or-v1-abcdefghij1234' → 'sk-or-v1...1234'
    """
    if len(key) > 12:
        return key[:8] + "..." + key[-4:]
    return "..." if len(key) <= 4 else key[:4] + "..."


# ── Key resolution ───────────────────────────────────────────────────────────

def resolve_api_key() -> str | None:
    """Return the best available API key, or None if none is configured.

    Priority: env var → config file.
    """
    env = os.environ.get(_ENV_VAR, "").strip()
    if env:
        return env
    stored = _read_config().get("openrouter_api_key", "").strip()
    return stored or None


def get_stored_key() -> str | None:
    """Return only the key from config.json (not env var).

    Used by /key show to display what is persisted on disk.
    """
    stored = _read_config().get("openrouter_api_key", "").strip()
    return stored or None


# ── Config file CRUD ─────────────────────────────────────────────────────────

def save_key(key: str) -> None:
    """Persist key to ~/.comet/config.json. Caller must validate first."""
    data = _read_config()
    data["openrouter_api_key"] = key
    _write_config(data)


def clear_key() -> None:
    """Remove the stored key from config.json (leaves other keys intact)."""
    data = _read_config()
    data.pop("openrouter_api_key", None)
    _write_config(data)


def config_file_path() -> Path:
    """Return the config file path (for display purposes)."""
    return _CONFIG_FILE


# ── Internal I/O ─────────────────────────────────────────────────────────────

def _read_config() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_config(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
