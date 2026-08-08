from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolArtifact:
    """A successful tool output that may be used to synthesize the final answer.

    ``kind`` is intentionally extensible.  The orchestration layer understands
    ``documents`` and ``memory`` as structured context; every other kind is
    passed to the generator as a verified tool output.
    """

    content: str
    kind: str = "tool_output"
    source_tool: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfirmedAction:
    """A completed side effect that the final answer may acknowledge.

    This is operation status, not factual evidence and never belongs in
    ``ToolArtifact``.
    """

    content: str
    source_tool: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    content: str                                        # 喂回模型的自然语言观察
    data: Dict[str, Any] = field(default_factory=dict)  # 编排层用的结构化载荷
    error: Optional[str] = None                         # ok=False 时的错误说明
    artifacts: List[ToolArtifact] = field(default_factory=list)  # 可进入终局合成的只读结果
    side_effects: List[Any] = field(default_factory=list)         # 写入/变更记录，不作为回答证据
    confirmed_actions: List[ConfirmedAction] = field(default_factory=list)  # 已完成操作状态


class Tool(Protocol):
    name: str
    description: str
    # arg name -> {"type": "str"|"int"|"number", "required": bool, "desc": str}
    args_schema: Dict[str, Dict[str, Any]]

    def run(self, args: Dict[str, Any]) -> ToolResult: ...
