"""Read-only repo tools with a shared internal definition layer."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from typing import get_args, get_origin

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

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


class PrintTreeArgs(BaseModel):
    path: str = "."
    depth: int = Field(default=2, ge=1, le=8)
    include_hidden: bool = False


def _print_tree(path: str = ".", depth: int = 2, include_hidden: bool = False) -> str:
    base = _resolve_repo_path(path)
    if not base.exists():
        return f"[error] path does not exist: {path}"
    if not base.is_dir():
        return f"[error] not a directory: {path}"

    root_parts_len = len(base.parts)
    lines: list[str] = [base.relative_to(PROJECT_ROOT).as_posix() or "."]

    for p in sorted(base.rglob("*")):
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        rel_parts = rel.split("/")
        if any(part in _IGNORED_DIR_NAMES for part in rel_parts):
            continue
        if not include_hidden and any(part.startswith(".") for part in rel_parts):
            continue
        level = len(p.parts) - root_parts_len
        if level > depth:
            continue
        indent = "  " * max(level, 0)
        suffix = "/" if p.is_dir() else ""
        lines.append(f"{indent}- {p.name}{suffix}")
        if len(lines) >= 300:
            lines.append("... [truncated]")
            break

    return "\n".join(lines)


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


class WriteFileArgs(BaseModel):
    path: str
    content: str
    create_dirs: bool = False


def _write_file(path: str, content: str, create_dirs: bool = False) -> str:
    file_path = _resolve_repo_path(path)
    parent = file_path.parent
    if not parent.exists():
        if not create_dirs:
            return f"[error] parent directory does not exist: {parent.relative_to(PROJECT_ROOT).as_posix()}"
        parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(content, encoding="utf-8")
    return f"[ok] wrote {file_path.relative_to(PROJECT_ROOT).as_posix()}"


class ReplaceTextArgs(BaseModel):
    path: str
    old_text: str = Field(min_length=1)
    new_text: str
    replace_all: bool = False


def _replace_text(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    file_path = _resolve_repo_path(path)
    if not file_path.exists():
        return f"[error] file does not exist: {path}"
    if not file_path.is_file():
        return f"[error] not a file: {path}"

    content = file_path.read_text(encoding="utf-8", errors="replace")
    occurrences = content.count(old_text)
    if occurrences == 0:
        return "[error] old_text not found"
    if not replace_all and occurrences > 1:
        return "[error] old_text matched multiple locations; refine the selection or use replace_all=true"

    if replace_all:
        updated = content.replace(old_text, new_text)
        replaced_count = occurrences
    else:
        updated = content.replace(old_text, new_text, 1)
        replaced_count = 1

    file_path.write_text(updated, encoding="utf-8")
    return f"[ok] replaced {replaced_count} occurrence(s) in {file_path.relative_to(PROJECT_ROOT).as_posix()}"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., str]
    requires_approval: bool = False


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
        name="print_tree",
        description="Print a shallow directory tree for quick structure inspection.",
        args_model=PrintTreeArgs,
        fn=_print_tree,
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
    ToolSpec(
        name="write_file",
        description="Create or overwrite a file with the provided content. Requires user approval.",
        args_model=WriteFileArgs,
        fn=_write_file,
        requires_approval=True,
    ),
    ToolSpec(
        name="replace_text",
        description="Replace exact text in an existing file. Requires user approval.",
        args_model=ReplaceTextArgs,
        fn=_replace_text,
        requires_approval=True,
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
    except ValidationError as exc:
        return _format_validation_error(tool_name, exc)
    except Exception as exc:
        return f"[error] tool execution failed: {exc}"


def get_langchain_tools(*, include_mutating: bool = True) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=spec.fn,
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_model,
        )
        for spec in TOOL_SPECS
        if include_mutating or not spec.requires_approval
    ]


def build_tool_schema_markdown(*, include_mutating: bool = True) -> str:
    lines = []
    for spec in TOOL_SPECS:
        if not include_mutating and spec.requires_approval:
            continue
        rendered_fields = ", ".join(
            _render_field_schema(name, field_info)
            for name, field_info in spec.args_model.model_fields.items()
        )
        approval = " Approval: required." if spec.requires_approval else ""
        lines.append(f"- `{spec.name}`: {spec.description}{approval} Args: {{{rendered_fields}}}")
    return "\n".join(lines)


def tool_requires_approval(tool_name: str) -> bool:
    spec = TOOL_SPECS_BY_NAME.get(tool_name)
    return bool(spec and spec.requires_approval)


def _format_validation_error(tool_name: str, exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors():
        path = ".".join(str(part) for part in error.get("loc", [])) or "argument"
        message = error.get("msg", "invalid value")
        details.append(f"{path}: {message}")
    joined = "; ".join(details) if details else str(exc)
    return f"[error] invalid args for {tool_name}: {joined}"


def _render_field_schema(name: str, field_info) -> str:
    parts = [f"{name}: {_render_annotation(field_info.annotation)}"]

    constraints: list[str] = []
    for metadata in field_info.metadata:
        if hasattr(metadata, "ge") and metadata.ge is not None:
            constraints.append(f">={metadata.ge}")
        if hasattr(metadata, "gt") and metadata.gt is not None:
            constraints.append(f">{metadata.gt}")
        if hasattr(metadata, "le") and metadata.le is not None:
            constraints.append(f"<={metadata.le}")
        if hasattr(metadata, "lt") and metadata.lt is not None:
            constraints.append(f"<{metadata.lt}")

    default = field_info.default
    if default is not None and str(default) != "PydanticUndefined":
        parts.append(f"default={default!r}")
    if constraints:
        parts.append("range=" + ",".join(constraints))
    return " ".join(parts)


def _render_annotation(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        if annotation is None:
            return "null"
        return getattr(annotation, "__name__", str(annotation))

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if origin in {list, tuple, set} and args:
        return f"{origin.__name__}[{_render_annotation(args[0])}]"
    if origin is dict and len(args) == 2:
        return f"dict[{_render_annotation(args[0])}, {_render_annotation(args[1])}]"
    if args:
        return " | ".join(_render_annotation(arg) for arg in args)
    return str(annotation)
