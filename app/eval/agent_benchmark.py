from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from app.agent.agent import MemoryAgent
from app.agent.loop import ToolAgent
from app.agent.registry import ToolRegistry
from app.agent.router import Router
from app.agent.tools.base import ConfirmedAction, ToolArtifact
from app.agent.tools.calculator import CalculatorTool
from app.agent.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from app.agent.tools.retrieve import RetrieveDocsTool
from app.agent.tools.workspace import make_workspace_tools
from app.memory.extractor import ExtractedFact
from app.memory.merger import MergeOp
from app.memory.models import MemoryFact


BENCHMARK_COHORTS = {"shared_core", "extended_tools", "robustness", "reserved"}
ANSWER_KINDS = {"exact", "contains_all", "labelled_number", "refusal"}
BASELINE_TOOLS = {"retrieve_docs", "read_memory", "write_memory"}
LOOP_TOOLS = BASELINE_TOOLS | {
    "calculator",
    "glob_files",
    "grep_text",
    "read_file",
}


def _normalise(text: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()


def _strip_fences(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value[3:]
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3]
    return value.strip()


def load_benchmark(path: str) -> Tuple[Dict[str, Any], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    validate_benchmark(data)
    return data, hashlib.sha256(raw).hexdigest()


def validate_benchmark(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict) or not str(data.get("benchmark_id", "")).strip():
        raise ValueError("benchmark_id is required")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")

    seen: set[str] = set()
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} must be an object")
        task_id = str(task.get("id", "")).strip()
        if not task_id or task_id in seen:
            raise ValueError(f"task {index}: missing or duplicate id={task_id!r}")
        seen.add(task_id)

        cohort = str(task.get("cohort", "")).strip()
        if cohort not in BENCHMARK_COHORTS:
            raise ValueError(f"{task_id}: invalid cohort={cohort!r}")
        enabled = bool(task.get("enabled", True))
        if not enabled:
            if cohort != "reserved" or not str(task.get("skip_reason", "")).strip():
                raise ValueError(f"{task_id}: disabled tasks must be reserved with skip_reason")
            continue

        if not str(task.get("question", "")).strip():
            raise ValueError(f"{task_id}: question is required")
        if not isinstance(task.get("baseline_route"), dict):
            raise ValueError(f"{task_id}: baseline_route is required")
        script = task.get("loop_script")
        if not isinstance(script, list) or not script:
            raise ValueError(f"{task_id}: loop_script is required")

        expected = task.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"{task_id}: expected is required")
        counts = expected.get("tool_counts")
        if not isinstance(counts, dict) or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()
        ):
            raise ValueError(f"{task_id}: expected.tool_counts must contain non-negative integers")
        answer = expected.get("answer")
        if not isinstance(answer, dict) or answer.get("kind") not in ANSWER_KINDS:
            raise ValueError(f"{task_id}: unsupported answer rule")
        if not str(answer.get("gold", "")).strip():
            raise ValueError(f"{task_id}: expected.answer.gold is required")

        doc_ids = [str(d.get("doc_id", "")).strip() for d in task.get("documents", [])]
        if any(not item for item in doc_ids) or len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"{task_id}: document ids must be non-empty and unique")

        workspace_files = task.get("workspace_files", {})
        if not isinstance(workspace_files, dict):
            raise ValueError(f"{task_id}: workspace_files must be an object")
        for raw_path, content in workspace_files.items():
            path = Path(str(raw_path))
            if (
                not str(raw_path).strip()
                or path.is_absolute()
                or ".." in path.parts
                or not isinstance(content, str)
            ):
                raise ValueError(f"{task_id}: invalid workspace fixture path/content")


def judge_answer(answer: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(rule.get("kind", ""))
    normalised = _normalise(answer)
    reasons: List[str] = []

    if kind == "exact":
        correct = normalised == _normalise(rule.get("value", rule.get("gold", "")))
        if not correct:
            reasons.append("normalised answer did not exactly match")
    elif kind == "contains_all":
        missing = [str(v) for v in rule.get("values", []) if _normalise(v) not in normalised]
        forbidden = [
            str(v) for v in rule.get("forbidden_values", []) if _normalise(v) in normalised
        ]
        correct = not missing and not forbidden
        reasons.extend(f"missing required text: {value}" for value in missing)
        reasons.extend(f"contained forbidden text: {value}" for value in forbidden)
    elif kind == "labelled_number":
        # Deliberately score only a labelled final result. Incidental numbers such as
        # "输入是 42，但答案不是 42" cannot pass this rule.
        matches = re.findall(
            r"(?:最终答案|final\s*answer)\s*[:：=]\s*(-?\d+(?:\.\d+)?)",
            unicodedata.normalize("NFKC", answer or ""),
            flags=re.IGNORECASE,
        )
        try:
            expected = Decimal(str(rule["value"]))
            actual = Decimal(matches[-1]) if matches else None
        except (InvalidOperation, KeyError):
            actual = None
            expected = Decimal("NaN")
        correct = actual is not None and actual == expected
        if not matches:
            reasons.append("missing explicit 最终答案 label")
        elif not correct:
            reasons.append(f"labelled final number was {matches[-1]}, expected {rule.get('value')}")
    elif kind == "refusal":
        markers = [str(v) for v in rule.get("markers", [])]
        forbidden = [
            str(v) for v in rule.get("forbidden_values", []) if _normalise(v) in normalised
        ]
        forbidden_patterns = [
            str(pattern)
            for pattern in rule.get("forbidden_patterns", [])
            if re.search(str(pattern), answer or "", flags=re.IGNORECASE)
        ]
        marker_ok = any(_normalise(marker) in normalised for marker in markers)
        correct = marker_ok and not forbidden and not forbidden_patterns
        if not marker_ok:
            reasons.append("missing an accepted insufficient-evidence marker")
        reasons.extend(f"contained forbidden factual claim: {value}" for value in forbidden)
        reasons.extend(
            f"matched forbidden factual pattern: {pattern}"
            for pattern in forbidden_patterns
        )
    else:
        raise ValueError(f"unsupported answer rule: {kind}")

    return {"correct": bool(correct), "kind": kind, "reasons": reasons}


def score_tool_calls(actual: Sequence[str], expected_counts: Dict[str, int]) -> Dict[str, Any]:
    actual_counts = Counter(actual)
    expected = Counter({str(k): int(v) for k, v in expected_counts.items() if int(v) > 0})
    true_positive = sum(min(actual_counts[name], expected[name]) for name in expected)
    actual_total = sum(actual_counts.values())
    expected_total = sum(expected.values())
    precision = true_positive / actual_total if actual_total else (1.0 if not expected else 0.0)
    recall = true_positive / expected_total if expected_total else (1.0 if not actual_counts else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "exact": actual_counts == expected,
        "actual_counts": dict(sorted(actual_counts.items())),
        "expected_counts": dict(sorted(expected.items())),
    }


class AuditMemoryStore:
    """Small deterministic in-memory store used to isolate orchestration from retrieval quality."""

    def __init__(self, initial: Sequence[Dict[str, Any]]) -> None:
        self._facts: Dict[str, MemoryFact] = {}
        for row in initial:
            fact = MemoryFact(
                id=str(row["id"]),
                fact_object=str(row.get("fact_object", "")),
                fact_content=str(row["fact_content"]),
                visibility=str(row.get("visibility", "PUBLIC")),
                source="benchmark",
            )
            self._facts[fact.id] = fact

    def add(self, fact: MemoryFact) -> MemoryFact:
        self._facts[fact.id] = fact
        return fact

    def list_active(self) -> List[MemoryFact]:
        return [f for f in self._facts.values() if f.state == "ACTIVE"]

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[MemoryFact, float]]:
        query_chars = set(_normalise(query))
        scored: List[Tuple[MemoryFact, float]] = []
        for fact in self.list_active():
            content_chars = set(_normalise(fact.fact_content))
            overlap = len(query_chars & content_chars)
            score = float(overlap + 1)  # keep fixtures recallable even after paraphrase
            if score > min_score:
                scored.append((fact, score))
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:top_k]


class AuditExtractor:
    def __init__(self, task: Dict[str, Any]) -> None:
        self.fact = task.get("write_fact")

    def extract(
        self, text: str, source: str = "chat", message_time: Optional[str] = None
    ) -> List[ExtractedFact]:
        del text, source, message_time
        if not isinstance(self.fact, dict):
            return []
        return [
            ExtractedFact(
                fact_object=str(self.fact.get("fact_object", "")),
                fact_content=str(self.fact["fact_content"]),
                visibility=str(self.fact.get("visibility", "PUBLIC")),
            )
        ]


class AuditMerger:
    def __init__(self, store: AuditMemoryStore) -> None:
        self.store = store

    def merge(self, facts: List[ExtractedFact], source: str = "chat") -> List[MergeOp]:
        existing = {_normalise(f.fact_content) for f in self.store.list_active()}
        ops: List[MergeOp] = []
        for fact in facts:
            if _normalise(fact.fact_content) in existing:
                ops.append(
                    MergeOp(
                        type="add",
                        fact_content=fact.fact_content,
                        fact_object=fact.fact_object,
                        visibility=fact.visibility,
                        applied=False,
                    )
                )
                continue
            self.store.add(
                MemoryFact(
                    fact_content=fact.fact_content,
                    fact_object=fact.fact_object,
                    visibility=fact.visibility,
                    source=source,
                )
            )
            existing.add(_normalise(fact.fact_content))
            ops.append(
                MergeOp(
                    type="add",
                    fact_content=fact.fact_content,
                    fact_object=fact.fact_object,
                    visibility=fact.visibility,
                    applied=True,
                )
            )
        return ops


def _chunk_from_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {
            "doc_id": doc["doc_id"],
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "text": doc["text"],
            "modality": "text",
        }
    }


class RecordingRetriever:
    def __init__(self, documents: Sequence[Dict[str, Any]]) -> None:
        self.documents = list(documents)
        self.queries: List[str] = []
        self.returned: List[Dict[str, Any]] = []

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        self.queries.append(query)
        q = _normalise(query)
        ranked: List[Tuple[int, int, Dict[str, Any]]] = []
        for index, doc in enumerate(self.documents):
            score = sum(
                1 for term in doc.get("match_terms", []) if _normalise(term) in q
            )
            if score > 0:
                ranked.append((-score, index, doc))
        rows = [_chunk_from_document(doc) for _score, _index, doc in sorted(ranked)]
        self.returned.extend(rows)
        return rows


class ScriptedComplete:
    def __init__(self, responses: Sequence[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        index = self.calls
        self.calls += 1
        if index >= len(self.responses):
            return json.dumps({"final_answer": True})
        response = self.responses[index]
        if set(response) == {"raw"}:
            return str(response["raw"])
        return json.dumps(response, ensure_ascii=False)


class ScriptedRouteComplete:
    def __init__(self, route: Dict[str, Any]) -> None:
        self.route = route
        self.calls = 0
        self.invalid_json = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        self.calls += 1
        return json.dumps(self.route, ensure_ascii=False)


class CountingComplete:
    def __init__(self, fn: Callable[[str, str], str], validate_json: bool = False) -> None:
        self.fn = fn
        self.validate_json = validate_json
        self.calls = 0
        self.invalid_json = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        raw = self.fn(system_prompt, user_prompt)
        if self.validate_json:
            try:
                if not isinstance(json.loads(_strip_fences(raw)), dict):
                    raise ValueError("not an object")
            except Exception:
                self.invalid_json += 1
        return raw


def _doc_id(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
    explicit = str(meta.get("doc_id", "")).strip()
    return explicit or f"{meta.get('source', 'unknown')}#{meta.get('chunk_id', '?')}"


def _calculation_results(artifacts: Sequence[ToolArtifact]) -> List[Any]:
    return [
        artifact.data.get("result")
        for artifact in artifacts
        if artifact.kind == "calculation" and "result" in artifact.data
    ]


def _required_calculations(expected: Dict[str, Any]) -> List[Any]:
    if "required_calculations" in expected:
        return list(expected["required_calculations"])
    if "required_calculation" in expected:
        return [expected["required_calculation"]]
    return []


def _decimal_multiset_contains(actual: Sequence[Any], required: Sequence[Any]) -> bool:
    try:
        actual_counts = Counter(Decimal(str(value)) for value in actual)
        required_counts = Counter(Decimal(str(value)) for value in required)
    except InvalidOperation:
        return False
    return all(actual_counts[value] >= count for value, count in required_counts.items())


def _artifact_text(artifacts: Sequence[ToolArtifact]) -> str:
    return "\n".join(artifact.content for artifact in artifacts)


def _evidence_ready(
    task: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    memory_text: str,
    artifacts: Sequence[ToolArtifact],
) -> bool:
    expected = task["expected"]
    actual_docs = {_doc_id(c) for c in chunks}
    docs_ok = set(expected.get("required_doc_ids", [])) <= actual_docs
    memory_ok = all(
        _normalise(value) in _normalise(memory_text)
        for value in expected.get("required_memory_contains", [])
    )
    calculations = _calculation_results(artifacts)
    required_calculations = _required_calculations(expected)
    calc_ok = _decimal_multiset_contains(calculations, required_calculations)
    artifact_text = _artifact_text(artifacts)
    artifact_ok = all(
        _normalise(value) in _normalise(artifact_text)
        for value in expected.get("required_artifact_contains", [])
    )
    no_docs_ok = not expected.get("expect_no_documents") or not actual_docs
    return docs_ok and memory_ok and calc_ok and artifact_ok and no_docs_ok


class OracleGenerator:
    """Deterministic evidence gate for regression tests, never an effectiveness judge."""

    def __init__(self, task: Dict[str, Any]) -> None:
        self.task = task
        self.calls = 0

    def __call__(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        memory_block: str,
        artifacts: Optional[List[ToolArtifact]] = None,
        confirmed_actions: Optional[List[ConfirmedAction]] = None,
    ) -> Dict[str, Any]:
        del query, confirmed_actions
        self.calls += 1
        evidence = artifacts or []
        if _evidence_ready(self.task, chunks, memory_block, evidence):
            answer = str(self.task["expected"]["answer"]["gold"])
        else:
            answer = "无法根据现有工具结果回答。"
        return {
            "answer": answer,
            "sources": [
                {"doc_id": _doc_id(c), "source": c.get("metadata", {}).get("source")}
                for c in chunks
            ],
        }


class RealGenerator:
    def __init__(self, complete_fn: Callable[[str, str], str]) -> None:
        self.complete_fn = complete_fn
        self.calls = 0

    def __call__(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        memory_block: str,
        artifacts: Optional[List[ToolArtifact]] = None,
        confirmed_actions: Optional[List[ConfirmedAction]] = None,
    ) -> Dict[str, Any]:
        self.calls += 1
        doc_text = "\n".join(
            f"[{_doc_id(chunk)}] {chunk.get('metadata', {}).get('text', '')}" for chunk in chunks
        )
        tool_text = "\n".join(
            f"[{artifact.source_tool}/{artifact.kind}] {artifact.content}"
            for artifact in (artifacts or [])
        )
        action_text = "\n".join(
            f"[{action.source_tool}] {action.content}" for action in (confirmed_actions or [])
        )
        system = (
            "你是离线 Agent 评测的答案合成器。只能使用给出的文档、记忆和已验证工具输出。"
            "“已确认动作”只用于确认操作是否完成，不是回答事实问题的证据。"
            "缺少回答所需证据时，明确回答“证据不足”。不要猜测。"
            "算术题必须在结尾单独写“最终答案：<数字>”。回答简短。"
        )
        prompt = (
            f"问题：\n{query}\n\n文档证据：\n{doc_text or '(无)'}\n\n"
            f"记忆证据：\n{memory_block or '(无)'}\n\n工具输出：\n{tool_text or '(无)'}\n\n"
            f"已确认动作：\n{action_text or '(无)'}"
        )
        answer = self.complete_fn(system, prompt)
        return {
            "answer": answer,
            "sources": [
                {"doc_id": _doc_id(c), "source": c.get("metadata", {}).get("source")}
                for c in chunks
            ],
        }


@dataclass
class AgentRun:
    answer: str
    called_tools: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    recalled_memories: List[MemoryFact]
    artifacts: List[ToolArtifact]
    confirmed_actions: List[ConfirmedAction]
    memory_after: List[MemoryFact]
    tool_calls: int
    failed_tool_calls: int
    steps: int
    parse_failures: int
    decision_calls: int
    generation_calls: int
    stop_reason: str


def _build_runtime_parts(task: Dict[str, Any]) -> Tuple[
    AuditMemoryStore, RecordingRetriever, AuditExtractor, AuditMerger
]:
    store = AuditMemoryStore(task.get("setup_memories", []))
    retriever = RecordingRetriever(task.get("documents", []))
    extractor = AuditExtractor(task)
    merger = AuditMerger(store)
    return store, retriever, extractor, merger


def _run_baseline(
    task: Dict[str, Any],
    mode: str,
    real_complete: Optional[Callable[[str, str], str]],
) -> AgentRun:
    store, retriever, extractor, merger = _build_runtime_parts(task)
    if mode == "scripted":
        route_complete: Any = ScriptedRouteComplete(task["baseline_route"])
        generator: Any = OracleGenerator(task)
    else:
        if real_complete is None:
            raise RuntimeError("real_complete is required in real mode")
        route_complete = CountingComplete(real_complete, validate_json=True)
        generator = RealGenerator(real_complete)

    agent = MemoryAgent(
        router=Router(complete_fn=route_complete),
        store=store,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        merger=merger,  # type: ignore[arg-type]
        retrieve_docs_fn=retriever,
        generate_fn=generator,
        message_time_fn=lambda: "2026-07-23",
        history_max_turns=0,
    )
    result = agent.chat(str(task["question"]))
    called: List[str] = []
    if result.route and result.route.use_memory:
        called.append("read_memory")
    if result.route and result.route.use_docs:
        called.append("retrieve_docs")
    if result.route and result.route.write_memory:
        called.append("write_memory")
    return AgentRun(
        answer=result.answer,
        called_tools=called,
        retrieved_chunks=list(retriever.returned),
        recalled_memories=list(result.recalled_memories),
        artifacts=[],
        confirmed_actions=[],
        memory_after=store.list_active(),
        tool_calls=len(called),
        failed_tool_calls=0,
        steps=1,
        parse_failures=int(getattr(route_complete, "invalid_json", 0)),
        decision_calls=int(route_complete.calls),
        generation_calls=int(generator.calls),
        stop_reason="single_pass",
    )


def _run_loop(
    task: Dict[str, Any],
    mode: str,
    real_complete: Optional[Callable[[str, str], str]],
) -> AgentRun:
    store, retriever, extractor, merger = _build_runtime_parts(task)
    registry = ToolRegistry()
    registry.register(RetrieveDocsTool(retrieve_docs_fn=retriever))
    registry.register(ReadMemoryTool(store=store))  # type: ignore[arg-type]
    registry.register(
        WriteMemoryTool(
            extractor=extractor,  # type: ignore[arg-type]
            merger=merger,  # type: ignore[arg-type]
            message_time_fn=lambda: "2026-07-23",
        )
    )
    registry.register(CalculatorTool())
    workspace_tmp: Optional[TemporaryDirectory[str]] = None
    workspace_files = task.get("workspace_files", {})
    if workspace_files:
        workspace_tmp = TemporaryDirectory(prefix="personal-rag-agent-benchmark-")
        workspace_root = Path(workspace_tmp.name)
        for relative, content in workspace_files.items():
            fixture_path = workspace_root / str(relative)
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(str(content), encoding="utf-8")
        for tool in make_workspace_tools(workspace_root):
            registry.register(tool)

    if mode == "scripted":
        controller: Any = ScriptedComplete(task["loop_script"])
        generator: Any = OracleGenerator(task)
    else:
        if real_complete is None:
            raise RuntimeError("real_complete is required in real mode")
        controller = CountingComplete(real_complete)
        generator = RealGenerator(real_complete)

    agent = ToolAgent(
        complete_fn=controller,
        registry=registry,
        generate_fn=generator,
        max_steps=8,
        max_tool_calls=8,
        history_max_turns=0,
    )
    try:
        result = agent.chat(str(task["question"]))
    finally:
        if workspace_tmp is not None:
            workspace_tmp.cleanup()
    called = [
        str(step.tool)
        for step in result.trajectory
        if step.tool and step.error != "duplicate call"
    ]
    artifacts = list(getattr(result, "tool_artifacts", []))
    confirmed_actions = list(getattr(result, "confirmed_actions", []))
    return AgentRun(
        answer=result.answer,
        called_tools=called,
        retrieved_chunks=list(retriever.returned),
        recalled_memories=list(result.recalled_memories),
        artifacts=artifacts,
        confirmed_actions=confirmed_actions,
        memory_after=store.list_active(),
        tool_calls=len(called),
        failed_tool_calls=sum(
            1 for step in result.trajectory if step.tool and not step.ok
        ),
        steps=len(result.trajectory),
        parse_failures=sum(
            1 for step in result.trajectory if step.error == "invalid decision JSON"
        ),
        decision_calls=int(controller.calls),
        generation_calls=int(generator.calls),
        stop_reason=result.stop_reason,
    )


def _score_evidence(task: Dict[str, Any], run: AgentRun) -> Dict[str, Any]:
    expected = task["expected"]
    actual_doc_ids = list(dict.fromkeys(_doc_id(chunk) for chunk in run.retrieved_chunks))
    memory_values = [fact.fact_content for fact in run.recalled_memories]
    calculations = _calculation_results(run.artifacts)
    required_docs = list(expected.get("required_doc_ids", []))
    required_memories = list(expected.get("required_memory_contains", []))
    required_calculations = _required_calculations(expected)
    required_artifact_text = list(expected.get("required_artifact_contains", []))

    docs_ok = set(required_docs) <= set(actual_doc_ids)
    memory_ok = all(
        any(_normalise(value) in _normalise(content) for content in memory_values)
        for value in required_memories
    )
    calc_ok = _decimal_multiset_contains(calculations, required_calculations)
    artifact_text = _artifact_text(run.artifacts)
    artifact_ok = all(
        _normalise(value) in _normalise(artifact_text)
        for value in required_artifact_text
    )
    no_docs_ok = not expected.get("expect_no_documents") or not actual_doc_ids
    return {
        "correct": docs_ok and memory_ok and calc_ok and artifact_ok and no_docs_ok,
        "required_doc_ids": required_docs,
        "actual_doc_ids": actual_doc_ids,
        "required_memory_contains": required_memories,
        "actual_memory_contents": memory_values,
        "required_calculations": required_calculations,
        "actual_calculations": calculations,
        "required_artifact_contains": required_artifact_text,
        "actual_artifact_sources": [
            artifact.source_tool for artifact in run.artifacts
        ],
        "expect_no_documents": bool(expected.get("expect_no_documents", False)),
    }


def _score_memory_state(task: Dict[str, Any], run: AgentRun) -> Dict[str, Any]:
    required = list(task["expected"].get("memory_after_contains", []))
    actual = [fact.fact_content for fact in run.memory_after]
    contents_ok = all(
        any(_normalise(value) in _normalise(content) for content in actual)
        for value in required
    )
    expected_count = task["expected"].get("memory_after_count")
    count_ok = expected_count is None or len(actual) == int(expected_count)
    return {
        "correct": contents_ok and count_ok,
        "required_contains": required,
        "expected_count": expected_count,
        "actual_count": len(actual),
        "actual_contents": actual,
    }


def score_run(task: Dict[str, Any], agent_name: str, run: AgentRun) -> Dict[str, Any]:
    expected = task["expected"]
    expected_counts = expected.get("tool_counts_by_agent", {}).get(
        agent_name, expected["tool_counts"]
    )
    tool_score = score_tool_calls(run.called_tools, expected_counts)
    expected_sequence = expected.get("tool_sequence_by_agent", {}).get(
        agent_name, expected.get("tool_sequence")
    )
    sequence_exact = (
        run.called_tools == list(expected_sequence)
        if isinstance(expected_sequence, list)
        else None
    )
    tool_score["expected_sequence"] = expected_sequence
    tool_score["actual_sequence"] = run.called_tools
    tool_score["sequence_exact"] = sequence_exact
    answer_score = judge_answer(run.answer, expected["answer"])
    evidence_score = _score_evidence(task, run)
    state_score = _score_memory_state(task, run)
    expected_failed = expected.get("failed_tool_calls")
    expected_parse = expected.get("parse_failures")
    recovery_score = {
        "correct": (
            (expected_failed is None or run.failed_tool_calls == int(expected_failed))
            and (expected_parse is None or run.parse_failures == int(expected_parse))
        ),
        "expected_failed_tool_calls": expected_failed,
        "actual_failed_tool_calls": run.failed_tool_calls,
        "expected_parse_failures": expected_parse,
        "actual_parse_failures": run.parse_failures,
        "recovered_to_final_answer": bool(
            (run.failed_tool_calls or run.parse_failures)
            and run.stop_reason == "final_answer"
        ),
    }
    available = BASELINE_TOOLS if agent_name == "baseline" else LOOP_TOOLS
    expected_tool_names = {name for name, count in expected_counts.items() if count > 0}
    supported = expected_tool_names & available
    capability_coverage = (
        len(supported) / len(expected_tool_names) if expected_tool_names else 1.0
    )
    success = bool(
        answer_score["correct"]
        and tool_score["exact"]
        and sequence_exact is not False
        and evidence_score["correct"]
        and state_score["correct"]
        and recovery_score["correct"]
    )
    return {
        "task_id": task["id"],
        "category": task["category"],
        "cohort": task["cohort"],
        "success": success,
        "answer": run.answer,
        "answer_judge": answer_score,
        "tools": tool_score,
        "evidence": evidence_score,
        "memory_state": state_score,
        "recovery": recovery_score,
        "capability_coverage": capability_coverage,
        "unsupported_expected_tools": sorted(expected_tool_names - available),
        "tool_calls": run.tool_calls,
        "failed_tool_calls": run.failed_tool_calls,
        "steps": run.steps,
        "parse_failures": run.parse_failures,
        "decision_calls": run.decision_calls,
        "generation_calls": run.generation_calls,
        "stop_reason": run.stop_reason,
        "refusal_case": task["expected"]["answer"]["kind"] == "refusal",
    }


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    refusal = [record for record in records if record["refusal_case"]]
    state_cases = [
        record for record in records if record["memory_state"]["required_contains"]
        or record["memory_state"]["expected_count"] is not None
    ]
    sequence_cases = [
        record for record in records
        if record["tools"]["sequence_exact"] is not None
    ]
    recovery_cases = [
        record for record in records
        if record["recovery"]["expected_failed_tool_calls"] is not None
        or record["recovery"]["expected_parse_failures"] is not None
    ]
    total_steps = sum(int(record["steps"]) for record in records)
    parse_failures = sum(int(record["parse_failures"]) for record in records)
    return {
        "n_tasks": len(records),
        "task_success": _mean(float(record["success"]) for record in records),
        "answer_accuracy": _mean(
            float(record["answer_judge"]["correct"]) for record in records
        ),
        "evidence_accuracy": _mean(
            float(record["evidence"]["correct"]) for record in records
        ),
        "tool_call_precision": _mean(
            float(record["tools"]["precision"]) for record in records
        ),
        "tool_call_recall": _mean(
            float(record["tools"]["recall"]) for record in records
        ),
        "tool_multiset_exact_match": _mean(
            float(record["tools"]["exact"]) for record in records
        ),
        "tool_sequence_exact_match": (
            _mean(float(record["tools"]["sequence_exact"]) for record in sequence_cases)
            if sequence_cases
            else None
        ),
        "n_tool_sequence_tasks": len(sequence_cases),
        "capability_coverage": _mean(
            float(record["capability_coverage"]) for record in records
        ),
        "memory_state_accuracy": (
            _mean(float(record["memory_state"]["correct"]) for record in state_cases)
            if state_cases
            else None
        ),
        "n_memory_state_tasks": len(state_cases),
        "refusal_accuracy": (
            _mean(float(record["answer_judge"]["correct"]) for record in refusal)
            if refusal
            else None
        ),
        "n_refusal_tasks": len(refusal),
        "avg_tool_calls": _mean(float(record["tool_calls"]) for record in records),
        "avg_failed_tool_calls": _mean(
            float(record["failed_tool_calls"]) for record in records
        ),
        "avg_steps": _mean(float(record["steps"]) for record in records),
        "avg_decision_calls": _mean(
            float(record["decision_calls"]) for record in records
        ),
        "parse_failure_rate": parse_failures / total_steps if total_steps else 0.0,
        "recovery_accuracy": (
            _mean(float(record["recovery"]["correct"]) for record in recovery_cases)
            if recovery_cases
            else None
        ),
        "n_recovery_tasks": len(recovery_cases),
        "stop_reason_counts": dict(
            sorted(Counter(str(record["stop_reason"]) for record in records).items())
        ),
    }


def detect_real_environment() -> Dict[str, bool]:
    from app.config import get_settings

    settings = get_settings()
    key = settings.openai_compatible_api_key.strip()
    base = settings.openai_compatible_base_url.strip()
    model = settings.llm_model.strip()
    key_ok = bool(key and key not in {"your_api_key", "sk-..."})
    return {
        "api_key_configured": key_ok,
        "base_url_configured": bool(base),
        "llm_model_configured": bool(model),
        "ready_for_real_mode": bool(key_ok and base and model),
    }


def run_benchmark(
    dataset_path: str,
    *,
    mode: str = "scripted",
    agents: Sequence[str] = ("baseline", "loop"),
    model: str = "",
) -> Dict[str, Any]:
    if mode not in {"scripted", "real"}:
        raise ValueError("mode must be scripted or real")
    if any(agent not in {"baseline", "loop"} for agent in agents):
        raise ValueError("agents must contain baseline and/or loop")

    data, dataset_sha = load_benchmark(dataset_path)
    enabled = [task for task in data["tasks"] if task.get("enabled", True)]
    reserved = [
        {
            "task_id": task["id"],
            "category": task["category"],
            "skip_reason": task["skip_reason"],
        }
        for task in data["tasks"]
        if not task.get("enabled", True)
    ]

    real_complete: Optional[Callable[[str, str], str]] = None
    environment = detect_real_environment()
    resolved_model = model
    if mode == "real":
        if not environment["ready_for_real_mode"]:
            raise RuntimeError("real mode requires configured API key, base URL, and LLM model")
        from app.agent.llm import make_complete_fn
        from app.config import get_settings

        resolved_model = model.strip() or get_settings().llm_model
        real_complete = make_complete_fn(model=resolved_model, temperature=0.0)

    output: Dict[str, Any] = {
        "benchmark_id": data["benchmark_id"],
        "benchmark_version": data["version"],
        "dataset_sha256": dataset_sha,
        "mode": mode,
        "result_usage": (
            "regression_only_not_an_effectiveness_claim"
            if mode == "scripted"
            else "real_model_effectiveness_run"
        ),
        "model": resolved_model if mode == "real" else "scripted_fixture",
        "real_environment": environment,
        "cohort_policy": {
            "main_comparison": "shared_core only",
            "extended_capabilities": "reported separately; excluded from main comparison",
            "robustness": "reported separately; excluded from main comparison",
        },
        "reserved_tasks": reserved,
        "agents": {},
    }

    for agent_name in agents:
        records: List[Dict[str, Any]] = []
        for task in enabled:
            run = (
                _run_baseline(task, mode, real_complete)
                if agent_name == "baseline"
                else _run_loop(task, mode, real_complete)
            )
            records.append(score_run(task, agent_name, run))

        shared = [record for record in records if record["cohort"] == "shared_core"]
        extended = [record for record in records if record["cohort"] == "extended_tools"]
        robustness = [record for record in records if record["cohort"] == "robustness"]
        output["agents"][agent_name] = {
            "main_comparison": summarise(shared),
            "extended_capabilities": summarise(extended),
            "robustness": summarise(robustness),
            "per_task": records,
        }

    if {"baseline", "loop"} <= set(output["agents"]):
        base = output["agents"]["baseline"]["main_comparison"]
        loop = output["agents"]["loop"]["main_comparison"]
        output["main_comparison_delta_loop_minus_baseline"] = {
            key: float(loop[key]) - float(base[key])
            for key in (
                "task_success",
                "answer_accuracy",
                "evidence_accuracy",
                "tool_call_precision",
                "tool_call_recall",
                "tool_multiset_exact_match",
                "avg_tool_calls",
                "avg_steps",
                "parse_failure_rate",
            )
        }
    return output
