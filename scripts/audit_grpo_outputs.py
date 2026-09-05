"""Verify completed GRPO checkpoints and their final adapter without generation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_rl.artifacts import sha256_file
from app.agent_rl.run_journal import atomic_json


def assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite(child)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file

    root = Path(args.run_root) / "results/e2_grpo"
    output = root / "integrity_audit.json"
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((root / "train_report.json").read_text())
    training = report["training"]
    assert training["completed_updates"] == len(training["updates"]) == 20
    assert training["episodes"] == 160 and training["training_pool_tasks"] == 128
    policy = report["policy"]
    assert policy["reference_unchanged"]
    assert policy["reference_state_sha256_before"] == policy["reference_state_sha256_after"]
    assert_finite(training)
    checks, protocols = [], set()
    for update in range(1, 21):
        directory = root / f"resume-update-{update:04d}"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["metadata"]["completed_updates"] == update
        assert manifest["metadata"]["total_episodes"] == update * 8
        digest = sha256_file(directory / "state.pt")
        assert digest == manifest["sha256"]
        protocols.add(manifest["metadata"]["protocol_sha256"])
        checks.append({"update": update, "state_sha256": digest})
    assert len(protocols) == 1
    state = torch.load(root / "resume-update-0020/state.pt", map_location="cpu", weights_only=True)
    saved = load_file(root / "adapter/policy/adapter_model.safetensors", device="cpu")
    mapping = {name.replace(".policy.", "."): value for name, value in state["policy"].items()}
    assert set(mapping) == set(saved)
    assert all(torch.equal(mapping[name], saved[name]) for name in saved)
    assert all(torch.isfinite(value).all() for value in saved.values())
    assert state["optimizer"] and state["rng"]
    audit = {"completed_updates": 20, "episodes": 160, "all_training_metrics_finite": True,
        "frozen_reference_unchanged": True, "all_complete_checkpoint_hashes_valid": True,
        "checkpoint_protocol_identical": True, "final_policy_equals_update20_checkpoint": True,
        "final_adapter_tensor_count": len(saved),
        "final_adapter_sha256": sha256_file(root / "adapter/policy/adapter_model.safetensors"),
        "checkpoints": checks}
    atomic_json(output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
