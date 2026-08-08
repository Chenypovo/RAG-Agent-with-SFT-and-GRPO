from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from app.agent.tools.base import Tool, ToolArtifact, ToolResult


_DEFAULT_IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules"}


class WorkspacePathError(ValueError):
    pass


@dataclass
class WorkspaceAccess:
    """Resolve paths and glob patterns without allowing access outside one root."""

    root: Path
    ignored_dirs: frozenset[str] = frozenset(_DEFAULT_IGNORED_DIRS)

    def __init__(self, root: Union[str, Path], ignored_dirs: Optional[Iterable[str]] = None) -> None:
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace root is not a directory: {resolved}")
        self.root = resolved
        self.ignored_dirs = frozenset(ignored_dirs or _DEFAULT_IGNORED_DIRS)

    def resolve_file(self, raw_path: str) -> Path:
        relative = self._validate_relative(raw_path, label="path")
        try:
            resolved = (self.root / relative).resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspacePathError(f"file not found: {raw_path}") from exc
        self._ensure_inside(resolved)
        if not resolved.is_file():
            raise WorkspacePathError(f"not a file: {raw_path}")
        return resolved

    def glob_files(self, raw_pattern: str, limit: Optional[int] = None) -> List[Path]:
        pattern = self._validate_relative(raw_pattern, label="pattern")
        matches: List[Path] = []
        pattern_parts = pattern.parts
        cap = max(int(limit), 1) if limit is not None else None
        for current, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in self.ignored_dirs)
            for filename in sorted(files):
                candidate = Path(current) / filename
                relative = candidate.relative_to(self.root)
                if not _glob_parts_match(relative.parts, pattern_parts):
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                    self._ensure_inside(resolved)
                except (FileNotFoundError, WorkspacePathError):
                    continue
                if resolved.is_file():
                    matches.append(resolved)
                    if cap is not None and len(matches) >= cap:
                        return matches
        return matches

    def relative(self, path: Path) -> str:
        self._ensure_inside(path)
        return path.relative_to(self.root).as_posix()

    def _validate_relative(self, raw_value: str, label: str) -> Path:
        value = str(raw_value or "").strip()
        if not value:
            raise WorkspacePathError(f"empty {label}")
        if "\x00" in value or len(value) > 1_024:
            raise WorkspacePathError(f"invalid {label}")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise WorkspacePathError(f"{label} must stay inside the workspace")
        if any(part in self.ignored_dirs for part in path.parts):
            raise WorkspacePathError(f"{label} is excluded from workspace tools")
        return path

    def _ensure_inside(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise WorkspacePathError("path escapes the workspace") from exc

def _glob_parts_match(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    @lru_cache(maxsize=None)
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index >= len(pattern_parts):
            return path_index >= len(path_parts)
        head = pattern_parts[pattern_index]
        if head == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], head)
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)


def _capped_text(lines: Iterable[str], max_chars: int) -> tuple[str, bool]:
    kept: List[str] = []
    used = 0
    truncated = False
    for line in lines:
        text = str(line)
        extra = len(text) + (1 if kept else 0)
        if used + extra > max_chars:
            remaining = max_chars - used - (1 if kept else 0)
            if remaining > 0:
                kept.append(text[:remaining])
            truncated = True
            break
        kept.append(text)
        used += extra
    return "\n".join(kept), truncated


def _with_truncation_notice(text: str, truncated: bool, max_chars: int, notice: str) -> str:
    if not truncated:
        return text[:max_chars]
    if len(notice) >= max_chars:
        return text[:max_chars]
    return text[: max_chars - len(notice)] + notice


class GlobFilesTool:
    name = "glob_files"
    description = "按 glob 模式列出 workspace 内的文件；只读，不能访问 workspace 外路径。"
    args_schema = {
        "pattern": {
            "type": "str",
            "required": True,
            "desc": "相对 workspace 的 glob，如 '**/*.py' 或 'app/**'",
        },
    }

    def __init__(self, workspace: WorkspaceAccess, max_results: int = 100, max_chars: int = 12_000) -> None:
        self.workspace = workspace
        self.max_results = max(int(max_results), 1)
        self.max_chars = max(int(max_chars), 1)

    def run(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern", ""))
        try:
            all_matches = self.workspace.glob_files(pattern, limit=self.max_results + 1)
        except WorkspacePathError as exc:
            return ToolResult(ok=False, content="", error=str(exc))

        selected = all_matches[: self.max_results]
        paths = [self.workspace.relative(path) for path in selected]
        chars_truncated = False
        if not paths:
            content = f"no files matched pattern: {pattern.strip()}"
        else:
            content, chars_truncated = _capped_text(paths, self.max_chars)
        truncated = len(all_matches) > len(selected) or (bool(paths) and chars_truncated)
        content = _with_truncation_notice(
            content,
            truncated,
            self.max_chars,
            "\n... results truncated",
        )
        return ToolResult(
            ok=True,
            content=content,
            data={"paths": paths, "truncated": truncated},
            artifacts=[
                ToolArtifact(
                    kind="workspace_files",
                    content=content,
                    source_tool=self.name,
                    data={"paths": paths},
                )
            ],
        )


class GrepTextTool:
    name = "grep_text"
    description = "在 workspace 文本文件中搜索字面文本；返回文件、行号和匹配行，只读。"
    args_schema = {
        "query": {"type": "str", "required": True, "desc": "要搜索的字面文本"},
        "pattern": {
            "type": "str",
            "required": False,
            "desc": "限制文件的 glob，默认 '**/*'",
        },
    }

    def __init__(
        self,
        workspace: WorkspaceAccess,
        max_results: int = 100,
        max_chars: int = 12_000,
        max_files_scanned: int = 1_000,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        self.workspace = workspace
        self.max_results = max(int(max_results), 1)
        self.max_chars = max(int(max_chars), 1)
        self.max_files_scanned = max(int(max_files_scanned), 1)
        self.max_file_bytes = max(int(max_file_bytes), 1)

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, content="", error="empty query")
        pattern = str(args.get("pattern") or "**/*").strip()
        try:
            files = self.workspace.glob_files(pattern, limit=self.max_files_scanned + 1)
        except WorkspacePathError as exc:
            return ToolResult(ok=False, content="", error=str(exc))

        matches: List[Dict[str, Any]] = []
        scanned = 0
        truncated = len(files) > self.max_files_scanned
        for path in files[: self.max_files_scanned]:
            if len(matches) >= self.max_results:
                truncated = True
                break
            scanned += 1
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query.casefold() not in line.casefold():
                    continue
                matches.append(
                    {
                        "path": self.workspace.relative(path),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                if len(matches) >= self.max_results:
                    truncated = True
                    break

        if not matches:
            content = f"no text matches for: {query}"
        else:
            rendered = (f"{m['path']}:{m['line']}: {m['text']}" for m in matches)
            content, chars_truncated = _capped_text(rendered, self.max_chars)
            truncated = truncated or chars_truncated
        content = _with_truncation_notice(
            content,
            truncated,
            self.max_chars,
            "\n... results truncated",
        )
        return ToolResult(
            ok=True,
            content=content,
            data={"matches": matches, "files_scanned": scanned, "truncated": truncated},
            artifacts=[
                ToolArtifact(
                    kind="workspace_search",
                    content=content,
                    source_tool=self.name,
                    data={"matches": matches},
                )
            ],
        )


class ReadFileTool:
    name = "read_file"
    description = "读取 workspace 内一个 UTF-8 文本文件；拒绝越界、目录和二进制文件。"
    args_schema = {
        "path": {"type": "str", "required": True, "desc": "相对 workspace 的文件路径"},
    }

    def __init__(
        self,
        workspace: WorkspaceAccess,
        max_chars: int = 12_000,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        self.workspace = workspace
        self.max_chars = max(int(max_chars), 1)
        self.max_file_bytes = max(int(max_file_bytes), 1)

    def run(self, args: Dict[str, Any]) -> ToolResult:
        raw_path = str(args.get("path", ""))
        try:
            path = self.workspace.resolve_file(raw_path)
        except WorkspacePathError as exc:
            return ToolResult(ok=False, content="", error=str(exc))
        try:
            with path.open("rb") as handle:
                raw = handle.read(self.max_file_bytes + 1)
        except OSError as exc:
            return ToolResult(ok=False, content="", error=f"file read failed: {exc}")
        if b"\x00" in raw:
            return ToolResult(ok=False, content="", error="binary files are not supported")

        byte_truncated = len(raw) > self.max_file_bytes
        text = raw[: self.max_file_bytes].decode("utf-8", errors="replace")
        relative = self.workspace.relative(path)
        prefix = f"{relative}:\n"
        body_max_chars = max(self.max_chars - len(prefix), 0)
        truncated = byte_truncated or len(text) > body_max_chars
        content = _with_truncation_notice(
            text,
            truncated,
            body_max_chars,
            "\n... file truncated",
        )
        rendered = (prefix + content)[: self.max_chars]
        return ToolResult(
            ok=True,
            content=rendered,
            data={"path": relative, "text": content, "truncated": truncated},
            artifacts=[
                ToolArtifact(
                    kind="file_content",
                    content=rendered,
                    source_tool=self.name,
                    data={"path": relative},
                )
            ],
        )


def make_workspace_tools(
    root: Union[str, Path],
    max_results: int = 100,
    max_chars: int = 12_000,
    max_files_scanned: int = 1_000,
    max_file_bytes: int = 1_000_000,
) -> List[Tool]:
    workspace = WorkspaceAccess(root)
    return [
        GlobFilesTool(workspace, max_results=max_results, max_chars=max_chars),
        GrepTextTool(
            workspace,
            max_results=max_results,
            max_chars=max_chars,
            max_files_scanned=max_files_scanned,
            max_file_bytes=max_file_bytes,
        ),
        ReadFileTool(workspace, max_chars=max_chars, max_file_bytes=max_file_bytes),
    ]
