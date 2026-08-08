from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent import factory
from app.agent.loop import ToolAgent
from app.agent.registry import ToolRegistry
from app.agent.tools.workspace import (
    GlobFilesTool,
    GrepTextTool,
    ReadFileTool,
    WorkspaceAccess,
    make_workspace_tools,
)


def make_workspace(tmp_path: Path) -> WorkspaceAccess:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Alpha\nneedle here\nOmega\n", encoding="utf-8")
    return WorkspaceAccess(tmp_path)


def test_glob_lists_workspace_files_with_artifact(tmp_path):
    tool = GlobFilesTool(make_workspace(tmp_path))
    result = tool.run({"pattern": "**/*.py"})

    assert result.ok
    assert result.data["paths"] == ["app/main.py"]
    assert result.artifacts[0].kind == "workspace_files"
    assert result.artifacts[0].source_tool == "glob_files"


def test_glob_empty_result_is_successful_and_bounded(tmp_path):
    tool = GlobFilesTool(make_workspace(tmp_path), max_chars=80)
    result = tool.run({"pattern": "**/*.rs"})

    assert result.ok
    assert result.data["paths"] == []
    assert "no files matched" in result.content
    assert len(result.content) <= 80


def test_glob_caps_result_count(tmp_path):
    workspace = make_workspace(tmp_path)
    for index in range(5):
        (tmp_path / f"{index}.txt").write_text(str(index), encoding="utf-8")
    result = GlobFilesTool(workspace, max_results=2).run({"pattern": "*.txt"})

    assert result.ok
    assert len(result.data["paths"]) == 2
    assert result.data["truncated"]
    assert "truncated" in result.content


def test_grep_searches_text_and_returns_line_number(tmp_path):
    result = GrepTextTool(make_workspace(tmp_path)).run(
        {"query": "NEEDLE", "pattern": "**/*.txt"}
    )

    assert result.ok
    assert result.data["matches"] == [
        {"path": "notes.txt", "line": 2, "text": "needle here"}
    ]
    assert "notes.txt:2" in result.content
    assert result.artifacts[0].kind == "workspace_search"


def test_grep_empty_result_is_successful(tmp_path):
    result = GrepTextTool(make_workspace(tmp_path)).run({"query": "missing"})
    assert result.ok
    assert result.data["matches"] == []
    assert "no text matches" in result.content


def test_read_file_returns_capped_text_artifact(tmp_path):
    result = ReadFileTool(make_workspace(tmp_path), max_chars=24).run({"path": "notes.txt"})

    assert result.ok
    assert result.data["path"] == "notes.txt"
    assert result.data["truncated"]
    assert result.artifacts[0].kind == "file_content"
    assert "Alpha" in result.content
    assert "Omega" not in result.content
    assert len(result.content) <= 24


@pytest.mark.parametrize(
    ("tool_factory", "args"),
    [
        (lambda ws: GlobFilesTool(ws), {"pattern": "../*"}),
        (lambda ws: GrepTextTool(ws), {"query": "secret", "pattern": "../*"}),
        (lambda ws: ReadFileTool(ws), {"path": "../secret.txt"}),
    ],
)
def test_workspace_tools_reject_parent_traversal(tmp_path, tool_factory, args):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    result = tool_factory(WorkspaceAccess(workspace_dir)).run(args)
    assert not result.ok
    assert "inside the workspace" in result.error


def test_read_file_rejects_absolute_path(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    result = ReadFileTool(make_workspace(tmp_path)).run({"path": str(outside)})
    assert not result.ok
    assert "inside the workspace" in result.error


def test_read_file_rejects_excluded_internal_directory(tmp_path):
    workspace = make_workspace(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    result = ReadFileTool(workspace).run({"path": ".git/config"})
    assert not result.ok
    assert "excluded" in result.error


def test_read_file_rejects_symlink_that_escapes_workspace(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    link = workspace_dir / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    result = ReadFileTool(WorkspaceAccess(workspace_dir)).run({"path": "link.txt"})
    assert not result.ok
    assert "escapes the workspace" in result.error


def test_workspace_artifact_reaches_final_generation(tmp_path):
    workspace = make_workspace(tmp_path)
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace))
    responses = iter(
        [
            '{"tool": "read_file", "args": {"path": "notes.txt"}}',
            '{"final_answer": true}',
        ]
    )
    captured = {}

    def complete(system_prompt, user_prompt):
        return next(responses)

    def generate(query, chunks, memory_block, artifacts):
        captured["artifacts"] = artifacts
        return {"answer": "used workspace evidence", "sources": []}

    result = ToolAgent(
        complete_fn=complete,
        registry=registry,
        generate_fn=generate,
    ).chat("notes 文件写了什么？")

    assert result.answer == "used workspace evidence"
    assert any(
        artifact.source_tool == "read_file" and "needle here" in artifact.content
        for artifact in captured["artifacts"]
    )


def test_workspace_tool_pack_contains_only_read_only_tools(tmp_path):
    names = {tool.name for tool in make_workspace_tools(tmp_path)}
    assert names == {"glob_files", "grep_text", "read_file"}


def test_factory_registers_workspace_tools_only_with_explicit_root(tmp_path, monkeypatch):
    runtime = SimpleNamespace(
        complete_fn=lambda system, user: '{"final_answer": true}',
        store=object(),
        vector_index=object(),
        index_json="unused.json",
        generate_fn=lambda query, chunks, memory, artifacts=None: {
            "answer": "",
            "sources": [],
        },
        retrieve=lambda query: [],
    )
    monkeypatch.setattr(factory, "_build_runtime", lambda *args, **kwargs: runtime)

    without_workspace = factory.build_tool_agent()
    with_workspace = factory.build_tool_agent(workspace_root=str(tmp_path))

    workspace_names = {"glob_files", "grep_text", "read_file"}
    assert workspace_names.isdisjoint(without_workspace.agent.registry.names())
    assert workspace_names.issubset(with_workspace.agent.registry.names())
