"""Freeze source/log provenance and SHA-256 manifests after all scientific work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(path, files, relative_to):
    total = 0
    with path.open("x") as handle:
        for source in files:
            label = str(source.relative_to(relative_to))
            if "\n" in label or "\r" in label or "\\" in label:
                raise ValueError("unsupported filename for portable checksum list")
            handle.write(digest(source) + "  " + label + "\n")
            total += source.stat().st_size
    return {"files": len(files), "bytes": total, "manifest_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    results = root / "results"
    workspace = root / "workspace"
    provenance = results / "delivery_provenance"
    manifest = results / "RESULTS_SHA256SUMS"
    required = ["final_integrity_audit.json", "paired_comparison.json", "paired_behaviour_comparison.json",
        "stopping_decision_audit.json", "stopping_decisions.jsonl", "ARTIFACTS.md"]
    assert all((results / name).is_file() for name in required)
    state = json.loads((results / "status.json").read_text())
    assert state["status"] == "experiments_complete_reporting_pending"
    assert len(state["completed_stages"]) == 13
    assert all(stage["status"] == "completed" for stage in state["completed_stages"])
    if provenance.exists() or manifest.exists():
        raise FileExistsError("delivery has already been frozen; preserve it and investigate")
    provenance.mkdir()
    frozen_source = provenance / "workspace"
    frozen_source.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", ".DS_Store")
    for name in ("app", "configs", "scripts", "tests"):
        shutil.copytree(workspace / name, frozen_source / name, ignore=ignore)
    shutil.copy2(workspace / "requirements.txt", frozen_source / "requirements.txt")
    shutil.copytree(root / "audit", provenance / "audit", ignore=shutil.ignore_patterns("e0", "__pycache__", "*.pyc"))
    e0 = provenance / "e0_history"
    e0.mkdir()
    shutil.copytree(root / "audit/e0/reports", e0 / "reports")
    shutil.copy2(root / "audit/e0/fixed_two_hop_audit.json", e0 / "fixed_two_hop_audit.json")
    documents = provenance / "documents"
    documents.mkdir()
    for name in ("ADAPTIVE_TEACHER_20260905_REPORT.md", "ADAPTIVE_TEACHER_20260905_AUDIT.md",
                 "ADAPTIVE_TEACHER_20260905_PILOT.md"):
        shutil.copy2(workspace / "docs" / name, documents / name)
    shutil.copy2(workspace / "docs/ADAPTIVE_TEACHER_20260905_REPORT.md", results / "REPORT.md")
    inputs = set(path for path in (root / "models").rglob("*") if path.is_file())
    for report_path in (results / "e1_teacher/report.json", results / "evaluation/prompt.json"):
        report = json.loads(report_path.read_text())
        data = Path(report["data_dir"])
        assert data.is_relative_to(root)
        inputs.update(path for path in data.rglob("*") if path.is_file())
    inputs.update([root / "audit/e0/agent-rl-overnight-2026-08-05.tar.gz", root / "audit/e0/teacher_train1617.jsonl"])
    external = write_manifest(provenance / "AUTODL_INPUTS_SHA256SUMS", sorted(inputs), root)
    metadata = {"frozen_at": datetime.now(timezone.utc).isoformat(), "run_root": str(root),
        "external_inputs": external, "scope": "All retained AutoDL result files and frozen provenance; base model weights, retrieval data/indexes and original E0 archive remain at their hashed AutoDL paths.",
        "source_policy": "Source snapshot includes final reporting scripts; original launch source hashes and the preserved diagnostic implementation are retained separately.",
        "checkpoint_policy": "SFT retains checkpoints550/568 under original save_total_limit=2; GRPO retains all20 complete update recovery states and final adapters.",
        "extra_local_audit_files": "Local initial-state and monitor copies may also exist outside this canonical remote results tree; frozen copies are under delivery_provenance/audit."}
    (provenance / "delivery.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = sorted(path for path in results.rglob("*") if path.is_file() and path != manifest)
    result = write_manifest(manifest, files, results)
    print(json.dumps({"results_manifest": str(manifest), **result, "external_inputs": external}, indent=2), flush=True)


if __name__ == "__main__":
    main()
