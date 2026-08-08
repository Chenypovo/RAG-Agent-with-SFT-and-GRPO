import hashlib
import json

import pytest

from app.agent_rl.artifacts import (
    ManifestValidationError,
    dependency_versions,
    ensure_outputs_available,
    sha256_file,
    validate_evaluation_inputs,
)
from app.agent_rl.hf_backend import TransformersChatBackend


def _write_manifest_dataset(tmp_path):
    contents = {
        "tasks.jsonl": b'{"task_id":"all"}\n',
        "tasks_test.jsonl": b'{"task_id":"test"}\n',
        "corpus.jsonl": b'{"source":"doc"}\n',
        "bm25.json": b'{"index":[]}\n',
    }
    for filename, content in contents.items():
        (tmp_path / filename).write_bytes(content)
    hashes = {
        filename: hashlib.sha256(content).hexdigest()
        for filename, content in contents.items()
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": hashes}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashes


def test_evaluation_inputs_match_manifest_and_report_hashes(tmp_path):
    expected = _write_manifest_dataset(tmp_path)

    provenance = validate_evaluation_inputs(tmp_path, task_filename="tasks_test.jsonl")

    assert provenance["data_manifest_sha256"] == sha256_file(tmp_path / "manifest.json")
    assert provenance["data_input_sha256"] == expected


def test_evaluation_inputs_reject_changed_file(tmp_path):
    _write_manifest_dataset(tmp_path)
    (tmp_path / "bm25.json").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="sha256 mismatch for bm25.json"):
        validate_evaluation_inputs(tmp_path, task_filename="tasks_test.jsonl")


def test_evaluation_inputs_require_every_declared_runtime_hash(tmp_path):
    _write_manifest_dataset(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"]["corpus.jsonl"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="missing a sha256 hash for corpus.jsonl"):
        validate_evaluation_inputs(tmp_path, task_filename="tasks_test.jsonl")


def test_existing_evaluation_outputs_require_explicit_overwrite(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        ensure_outputs_available((report, tmp_path / "trajectories.jsonl"), overwrite=False)
    ensure_outputs_available((report,), overwrite=True)


def test_dependency_versions_include_python_and_missing_packages():
    versions = dependency_versions(("definitely-not-a-real-package-name",))

    assert versions["python"]
    assert versions["packages"]["definitely-not-a-real-package-name"] is None


def test_transformers_backend_reports_resolved_model_commit():
    commit = "a" * 40

    class FakeConfig:
        _commit_hash = commit

    class FakeModel:
        config = FakeConfig()

        def eval(self):
            return self

    class FakeTokenizer:
        pass

    backend = TransformersChatBackend(
        "fake/model",
        revision="main",
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
    )

    assert backend.revision == "main"
    assert backend.resolved_commit == commit
