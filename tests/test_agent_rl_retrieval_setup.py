from pathlib import Path
import importlib.util
import sys

import pytest

from app.agent_rl import retrieval_setup


def test_hybrid_training_reuses_existing_registry_and_sentence_artifacts(tmp_path, monkeypatch):
    (tmp_path / "bm25.json").write_text("{}")
    (tmp_path / "lancedb").mkdir()
    (tmp_path / "lancedb" / "sentences").write_text("source#0")
    called = {}
    expected_registry = object()

    def build(**kwargs):
        called.update(kwargs)
        return expected_registry

    monkeypatch.setattr(retrieval_setup, "build_hybrid_registry", build)
    registry, provenance = retrieval_setup.build_retrieval_registry(tmp_path, {
        "backend": "hybrid-rerank", "embedding_model": "/models/bge-small-en-v1.5",
        "reranker_model": "/models/bge-reranker-base",
    })
    assert registry is expected_registry
    assert (called["candidate_top_k"], called["output_top_k"], called["rrf_k"]) == (15, 6, 60)
    assert called["vector_store"] == "lancedb"
    assert provenance["parent_child_expansion"] is False
    assert len(provenance["lancedb_sha256"]) == 64
    assert "parent_child" not in called


def test_hybrid_fails_before_model_loading_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval_setup, "build_hybrid_registry", lambda **kwargs: pytest.fail("loaded models"))
    with pytest.raises(FileNotFoundError, match="LanceDB"):
        retrieval_setup.build_retrieval_registry(tmp_path, {"backend": "hybrid-rerank"})


def test_sentence_bookkeeping_rejects_parent_expansion(tmp_path):
    with pytest.raises(ValueError, match="parent-child"):
        retrieval_setup.build_retrieval_registry(tmp_path, {
            "backend": "hybrid-rerank", "parent_child_expansion": True,
        })


def _teacher_script():
    path = Path(__file__).resolve().parents[1] / "scripts/build_agent_teacher_rollouts.py"
    spec = importlib.util.spec_from_file_location("teacher_cli_alignment_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extra", [
    ["--teacher-model", "/different/model"],
    ["--teacher-revision", "different-revision"],
    ["--candidates-per-task", "5"],
    ["--max-steps", "6"],
])
def test_teacher_rejects_unshared_model_or_excess_budget(monkeypatch, extra):
    module = _teacher_script()
    monkeypatch.setattr(sys, "argv", ["teacher", "--finalizer-model", "/models/7b", *extra])
    with pytest.raises(SystemExit):
        module.parse_args()


def test_scripted_default_remains_bm25_and_adaptive_default_is_hybrid(monkeypatch):
    module = _teacher_script()
    monkeypatch.setattr(sys, "argv", ["teacher", "--finalizer-model", "/models/7b"])
    assert module.parse_args().retrieval_backend == "hybrid-rerank"
    monkeypatch.setattr(sys, "argv", ["teacher", "--finalizer-model", "/models/7b", "--teacher-mode", "scripted"])
    assert module.parse_args().retrieval_backend == "bm25"
