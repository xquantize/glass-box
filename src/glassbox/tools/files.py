"""Sandboxed file tools — all paths are resolved under a workspace root."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from glassbox.tools.base import FunctionTool

DEFAULT_WORKSPACE = Path("workspace")
MAX_READ_BYTES = 100_000


def _resolve(root: Path, relative: str) -> Path:
    """Join ``relative`` under ``root``; reject path-traversal escapes."""
    if not relative or relative.strip() != relative:
        raise ValueError("path must be a non-empty relative path without surrounding whitespace")
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"path escapes workspace root: {relative!r}")
    return candidate


class ListFilesArgs(BaseModel):
    path: str = Field(
        default=".",
        description="Directory relative to the workspace root (default: '.').",
    )


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="Full text content to write to the file.")


def build_file_tools(root: str | Path = DEFAULT_WORKSPACE) -> list[FunctionTool]:
    """Return list_files / read_file / write_file bound to a workspace directory."""
    workspace = Path(root).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    def list_files(args: ListFilesArgs) -> str:
        target = _resolve(workspace, args.path)
        if not target.exists():
            raise FileNotFoundError(f"directory not found: {args.path}")
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {args.path}")
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return "(empty)"
        lines: list[str] = []
        for entry in entries:
            kind = "dir" if entry.is_dir() else "file"
            rel = entry.relative_to(workspace).as_posix()
            lines.append(f"{kind}\t{rel}")
        return "\n".join(lines)

    def read_file(args: ReadFileArgs) -> str:
        target = _resolve(workspace, args.path)
        if not target.exists():
            raise FileNotFoundError(f"file not found: {args.path}")
        if not target.is_file():
            raise IsADirectoryError(f"not a file: {args.path}")
        data = target.read_bytes()
        if len(data) > MAX_READ_BYTES:
            raise ValueError(
                f"file too large ({len(data)} bytes); max is {MAX_READ_BYTES}"
            )
        return data.decode("utf-8")

    def write_file(args: WriteFileArgs) -> str:
        target = _resolve(workspace, args.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        rel = target.relative_to(workspace).as_posix()
        return f"wrote {len(args.content)} characters to {rel}"

    return [
        FunctionTool(
            name="list_files",
            description=(
                f"List files and directories under the workspace root ({workspace}). "
                "Paths are relative to that root."
            ),
            parameters=ListFilesArgs,
            handler=list_files,
        ),
        FunctionTool(
            name="read_file",
            description=(
                f"Read a UTF-8 text file from the workspace root ({workspace}). "
                "Paths are relative to that root."
            ),
            parameters=ReadFileArgs,
            handler=read_file,
        ),
        FunctionTool(
            name="write_file",
            description=(
                f"Write a UTF-8 text file under the workspace root ({workspace}). "
                "Creates parent directories as needed. Paths are relative to that root."
            ),
            parameters=WriteFileArgs,
            handler=write_file,
        ),
    ]
