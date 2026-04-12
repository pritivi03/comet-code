"""Read-only repo tools with a shared internal definition layer."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path.cwd().resolve()
DEFAULT_MAX_CHARS = 20_000
_IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def _resolve_repo_path(path_value: str) -> Path:
    candidate = (PROJECT_ROOT / path_value).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {path_value}") from exc
    return candidate


def _truncate(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


class ListFilesArgs(BaseModel):
    path: str = "."
    limit: int = Field(default=200, ge=1, le=5_000)
    include_hidden: bool = False


def _list_files(path: str = ".", limit: int = 200, include_hidden: bool = False) -> str:
    base = _resolve_repo_path(path)
    if not base.exists():
        return f"[error] path does not exist: {path}"
    if not base.is_dir():
        return f"[error] not a directory: {path}"

    files: list[str] = []
    for file_path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = file_path.relative_to(PROJECT_ROOT).as_posix()

        parts = rel.split("/")
        if any(part in _IGNORED_DIR_NAMES for part in parts):
            continue
        if not include_hidden and any(part.startswith(".") for part in parts):
            continue
        files.append(rel)
        if len(files) >= limit:
            break

    if not files:
        return "[no files found]"
    return "\n".join(files)


class SearchTextArgs(BaseModel):
    pattern: str
    path: str = "."
    max_results: int = Field(default=200, ge=1, le=2_000)
    use_regex: bool = False


def _search_text(
    pattern: str,
    path: str = ".",
    max_results: int = 200,
    use_regex: bool = False,
) -> str:
    base = _resolve_repo_path(path)
    if not base.exists():
        return f"[error] path does not exist: {path}"

    cmd = [
        "rg",
        "-n",
        "--no-heading",
        "--color",
        "never",
        "--max-count",
        str(max_results),
    ]
    if not use_regex:
        cmd.append("-F")
    cmd.extend([
        pattern,
        str(base),
    ])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "[error] rg command not available"

    if proc.returncode not in (0, 1):
        stderr = proc.stderr.strip() or "unknown rg error"
        return f"[error] search failed: {stderr}"

    out = proc.stdout.strip()
    if not out:
        return "[no matches]"
    return _truncate(out)


class FindFilesArgs(BaseModel):
    pattern: str = "*.py"
    path: str = "."
    limit: int = Field(default=200, ge=1, le=5_000)
    include_hidden: bool = False


def _find_files(
    pattern: str = "*.py",
    path: str = ".",
    limit: int = 200,
    include_hidden: bool = False,
) -> str:
    base = _resolve_repo_path(path)
    if not base.exists():
        return f"[error] path does not exist: {path}"
    if not base.is_dir():
        return f"[error] not a directory: {path}"

    matches: list[str] = []
    for file_path in sorted(p for p in base.rglob(pattern) if p.is_file()):
        rel = file_path.relative_to(PROJECT_ROOT).as_posix()
        parts = rel.split("/")
        if any(part in _IGNORED_DIR_NAMES for part in parts):
            continue
        if not include_hidden and any(part.startswith(".") for part in parts):
            continue
        matches.append(rel)
        if len(matches) >= limit:
            break

    if not matches:
        return "[no matches]"
    return "\n".join(matches)


class ReadFileArgs(BaseModel):
    path: str
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=200, le=200_000)


def _read_file(path: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    file_path = _resolve_repo_path(path)
    if not file_path.exists():
        return f"[error] file does not exist: {path}"
    if not file_path.is_file():
        return f"[error] not a file: {path}"

    content = file_path.read_text(encoding="utf-8", errors="replace")
    return _truncate(content, max_chars=max_chars)


class ReadRangeArgs(BaseModel):
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=200, le=200_000)


def _read_range(
    path: str,
    start_line: int,
    end_line: int,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    if end_line < start_line:
        return "[error] end_line must be >= start_line"

    file_path = _resolve_repo_path(path)
    if not file_path.exists():
        return f"[error] file does not exist: {path}"
    if not file_path.is_file():
        return f"[error] not a file: {path}"

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start_index = start_line - 1
    end_index = min(end_line, len(lines))
    if start_index >= len(lines):
        return "[error] start_line beyond EOF"

    selected = lines[start_index:end_index]
    numbered = [
        f"{idx}: {line}"
        for idx, line in enumerate(selected, start=start_line)
    ]
    return _truncate("\n".join(numbered), max_chars=max_chars)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., str]


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="list_files",
        description="List repository files recursively under a path.",
        args_model=ListFilesArgs,
        fn=_list_files,
    ),
    ToolSpec(
        name="search_text",
        description="Search repository text using ripgrep and return matching lines.",
        args_model=SearchTextArgs,
        fn=_search_text,
    ),
    ToolSpec(
        name="find_files",
        description="Find files by glob pattern, e.g. ui.py or src/**/*.py.",
        args_model=FindFilesArgs,
        fn=_find_files,
    ),
    ToolSpec(
        name="read_file",
        description="Read a file from the repository.",
        args_model=ReadFileArgs,
        fn=_read_file,
    ),
    ToolSpec(
        name="read_range",
        description="Read a specific line range from a file.",
        args_model=ReadRangeArgs,
        fn=_read_range,
    ),
]

TOOL_SPECS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def execute_tool(tool_name: str, args: dict[str, Any]) -> str:
    spec = TOOL_SPECS_BY_NAME.get(tool_name)
    if spec is None:
        return f"[error] unknown tool: {tool_name}"
    try:
        parsed = spec.args_model.model_validate(args)
        return spec.fn(**parsed.model_dump())
    except Exception as exc:
        return f"[error] tool execution failed: {exc}"


def get_langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=spec.fn,
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_model,
        )
        for spec in TOOL_SPECS
    ]


def build_tool_schema_markdown() -> str:
    lines = []
    for spec in TOOL_SPECS:
        fields = ", ".join(spec.args_model.model_fields.keys())
        lines.append(f'- `{spec.name}`: {spec.description} Args: {{{fields}}}')
    return "\n".join(lines)
